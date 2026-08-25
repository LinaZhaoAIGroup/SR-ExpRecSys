from django.db import models
from django.utils import timezone


class Conversation(models.Model):
    """dialogue session model"""
    title = models.CharField(max_length=200, default="new conversation", verbose_name="Conversation title")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="creation time")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="Update time")

    class Meta:
        verbose_name = "dialogue session"
        verbose_name_plural = "dialogue session"
        ordering = ['-updated_at']

    def __str__(self):
        return self.title


class Message(models.Model):
    """message model"""
    ROLE_CHOICES = [
        ('user', 'User'),
        ('assistant', 'Assistant'),
    ]

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name="Belonging to session"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, verbose_name="role")
    content = models.TextField(verbose_name="Message content")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="creation time")
    
    # Store the user's original question (before intent recognition)
    original_question = models.TextField(blank=True, null=True, verbose_name="original question")

    class Meta:
        verbose_name = "news"
        verbose_name_plural = "news"
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"
