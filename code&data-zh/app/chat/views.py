from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
import logging

from .models import Conversation, Message
from .ai_service import intent, chat_with_history, wd, wd_kg_techniques, question_classifer, multilingual_output
from .graph_service import GraphRAGManager
from .vector_service import vector_silimar


logger = logging.getLogger(__name__)
graph_manager = GraphRAGManager.get_instance()


def index(request):
    """主页面，显示聊天界面"""
    conversations = Conversation.objects.all()[:20]  # 获取最近20个会话
    return render(request, 'chat/index.html', {
        'conversations': conversations
    })


@csrf_exempt
@require_http_methods(["POST"])
def create_conversation(request):
    """创建新会话"""
    try:
        conversation = Conversation.objects.create(title="新对话")
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
    """获取所有会话列表"""
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
    """获取指定会话的所有消息"""
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
    """发送消息并获取回复"""
    try:
        data = json.loads(request.body)
        conversation_id = data.get('conversation_id')
        user_message = data.get('message', '').strip()

        if not user_message:
            return JsonResponse({
                'success': False,
                'error': '消息内容不能为空'
            }, status=400)

        # 获取或创建会话
        if conversation_id:
            conversation = get_object_or_404(Conversation, id=conversation_id)
        else:
            conversation = Conversation.objects.create(title=user_message[:50])

        # 保存用户消息
        user_msg = Message.objects.create(
            conversation=conversation,
            role='user',
            content=user_message,
            original_question=user_message
        )

        # 意图识别
        #user_message_chinese = multilingual_input(user_message)
        print('user_message',user_message)
        recognized_intent = intent(user_message)
        print('recognized_intent', recognized_intent)
        # 获取历史消息（用于多轮对话）
        history_messages = conversation.messages.exclude(id=user_msg.id).values('role', 'content')
        print('history_messages', history_messages)
        history_list = list(history_messages)

        # 优先尝试技术知识图谱问答

        classifier = question_classifer(recognized_intent)
        print('问题判断：', classifier)
        retrieval_type = '未确定'
        retrieval_route = ''
        vector_score = None
        if 'yes' in classifier:
            assistant_response = wd_kg_techniques(user_message)
            retrieval_type = '技术知识库'
            retrieval_route = 'question_classifier=yes -> wd_kg_techniques'
            print("1 技术", assistant_response)

        if 'yes' not in classifier:
            assistant_response, score = vector_silimar(user_message)
            vector_score = float(score) if score is not None else None
            retrieval_type = '向量检索'
            retrieval_route = 'question_classifier!=yes -> vector_silimar'
            print('2 vector:',assistant_response)
            if score < 0.4:
                retrieval_type = '图检索'
                retrieval_route = 'question_classifier!=yes -> vector_silimar -> graph_manager.answer'
                assistant_response = graph_manager.answer(user_message, wd)
                # assistant_response = graph_rag.answer_question(recognized_intent)
                print("3 KG", assistant_response)
        #assistant_response = multilingual_output(user_message, assistant_response)



            # 如果图谱未命中，则根据是否存在历史消息选择模型
            if not assistant_response:
                if history_list:
                    retrieval_type = '历史对话回退'
                    retrieval_route = retrieval_route + ' -> chat_with_history'
                    assistant_response = chat_with_history(user_message, history_list)
                    print("4 history", assistant_response)
                else:
                    retrieval_type = '通用模型回退'
                    retrieval_route = retrieval_route + ' -> wd'
                    assistant_response = wd(user_message)
                    print("5 LLM", assistant_response)
            #assistant_response = multilingual_output(user_message,assistant_response)

        if assistant_response:
            assistant_response = multilingual_output(user_message, assistant_response)

        # 保存助手回复
        assistant_msg = Message.objects.create(
            conversation=conversation,
            role='assistant',
            content=assistant_response
        )

        # 更新会话标题（如果是新会话且标题还是默认的）
        if conversation.title == "新对话" and len(user_message) <= 50:
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
        logger.error(f"发送消息错误: {e}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def delete_conversation(request, conversation_id):
    """删除会话"""
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
    """更新会话标题"""
    try:
        data = json.loads(request.body)
        new_title = data.get('title', '').strip()

        if not new_title:
            return JsonResponse({
                'success': False,
                'error': '标题不能为空'
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
