from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging

from .models import Conversation, Message
from .ai_service import (
    chat_with_history,
    intent,
    multilingual_output,
    question_classifier,
    wd,
    wd_kg_techniques,
)
from .graph_service import GraphRAGManager
from .vector_service import vector_similarity


logger = logging.getLogger(__name__)
graph_manager = GraphRAGManager.get_instance()


def index(request):
    """The main page shows the chat interface"""
    conversations = Conversation.objects.all()[:20]  # Get the latest20sessions
    return render(request, 'chat/index.html', {
        'conversations': conversations
    })


@csrf_exempt
@require_http_methods(["POST"])
def create_conversation(request):
    """Create new session"""
    try:
        conversation = Conversation.objects.create(title="new conversation")
        return JsonResponse({
            'success': True,
            'conversation_id': conversation.id,
            'title': conversation.title
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_conversations(request):
    """Get a list of all sessions"""
    try:
        conversations = Conversation.objects.all().order_by('-updated_at')
        data = [{
            'id': conv.id,
            'title': conv.title,
            'created_at': conv.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': conv.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        } for conv in conversations]
        return JsonResponse({
            'success': True,
            'conversations': data
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["GET"])
def get_messages(request, conversation_id):
    """Get all messages of the specified conversation"""
    try:
        conversation = get_object_or_404(Conversation, id=conversation_id)
        messages = conversation.messages.all().order_by('created_at')
        data = [{
            'id': msg.id,
            'role': msg.role,
            'content': msg.content,
            'created_at': msg.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'original_question': msg.original_question
        } for msg in messages]
        return JsonResponse({
            'success': True,
            'messages': data
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def send_message(request):
    """Send a message and get a reply"""
    try:
        data = json.loads(request.body)
        conversation_id = data.get('conversation_id')
        user_message = data.get('message', '').strip()

        if not user_message:
            return JsonResponse({
                'success': False,
                'error': 'Message content cannot be empty'
            }, status=400)

        # Get or create a session
        if conversation_id:
            conversation = get_object_or_404(Conversation, id=conversation_id)
        else:
            conversation = Conversation.objects.create(title=user_message[:50])

        # Save user messages
        user_msg = Message.objects.create(
            conversation=conversation,
            role='user',
            content=user_message,
            original_question=user_message
        )

        # Intent recognition
        #user_message_chinese = multilingual_input(user_message)
        print('user_message',user_message)
        recognized_intent = intent(user_message)
        print('recognized_intent', recognized_intent)
        # Get historical messages (for multi-round conversations)
        history_messages = conversation.messages.exclude(id=user_msg.id).values('role', 'content')
        print('history_messages', history_messages)
        history_list = list(history_messages)

        # Prioritize trying the technical knowledge graph Q&A

        classifier = question_classifier(recognized_intent)
        print('Problem judgment:', classifier)
        retrieval_type = 'Undetermined'
        retrieval_route = ''
        vector_score = None
        if 'yes' in classifier:
            assistant_response = wd_kg_techniques(user_message)
            retrieval_type = 'Technical knowledge base'
            retrieval_route = 'question_classifier=yes -> wd_kg_techniques'
            print("1 technology", assistant_response)

        if 'yes' not in classifier:
            assistant_response, score = vector_similarity(user_message)
            vector_score = float(score) if score is not None else None
            retrieval_type = 'vector search'
            retrieval_route = 'question_classifier!=yes -> vector_similarity'
            print('2 vector:',assistant_response)
            # A value similarity below 0.4 is an explicit abstention state.
            # Do not route the same question to another generator afterward.
            if score < 0.4:
                retrieval_type = 'No reliable corpus match'
                retrieval_route = 'question_classifier!=yes -> vector_similarity -> abstain'
                print('No reliable corpus match; recommendation generation stopped.')
        #assistant_response = multilingual_output(user_message, assistant_response)



            # If the graph misses, the model is selected based on the presence of historical messages
            if not assistant_response:
                if history_list:
                    retrieval_type = 'Historical conversation rollback'
                    retrieval_route = retrieval_route + ' -> chat_with_history'
                    assistant_response = chat_with_history(user_message, history_list)
                    print("4 history", assistant_response)
                else:
                    retrieval_type = 'Universal model fallback'
                    retrieval_route = retrieval_route + ' -> wd'
                    assistant_response = wd(user_message)
                    print("5 LLM", assistant_response)
            #assistant_response = multilingual_output(user_message,assistant_response)

        if assistant_response:
            assistant_response = multilingual_output(user_message, assistant_response)

        # Save Assistant Reply
        assistant_msg = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=assistant_response
        )

        # Update session title (if this is a new session and the title is still the default)
        if conversation.title == "new conversation" and len(user_message) <= 50:
            conversation.title = user_message
            conversation.save()


        return JsonResponse({
            'success': True,
            'conversation_id': conversation.id,
            'user_message': {
                'id': user_msg.id,
                'content': user_message,
                'created_at': user_msg.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'original_question': user_message
            },
            'assistant_message': {
                'id': assistant_msg.id,
                'content': assistant_response,
                'created_at': assistant_msg.created_at.strftime('%Y-%m-%d %H:%M:%S')
            },
            'recognized_intent': recognized_intent,
            'classifier_result': classifier,
            'retrieval_type': retrieval_type,
            'retrieval_route': retrieval_route,
            'vector_similarity': vector_score,
            'vector_threshold': 0.4
        })
    except Exception as e:
        logger.error(f"Send message error: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_conversation(request, conversation_id):
    """Delete session"""
    try:
        conversation = get_object_or_404(Conversation, id=conversation_id)
        conversation.delete()
        return JsonResponse({
            'success': True
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def update_conversation_title(request, conversation_id):
    """Update session title"""
    try:
        data = json.loads(request.body)
        new_title = data.get('title', '').strip()

        if not new_title:
            return JsonResponse({
                'success': False,
                'error': 'Title cannot be empty'
            }, status=400)

        conversation = get_object_or_404(Conversation, id=conversation_id)
        conversation.title = new_title
        conversation.save()

        return JsonResponse({
            'success': True,
            'title': conversation.title
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)
