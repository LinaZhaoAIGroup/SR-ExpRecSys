from django.db import models
from django.utils import timezone


class Conversation(models.Model):
    """对话会话模型"""
    title = models.CharField(max_length=200, default="新对话", verbose_name="对话标题")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    updated_at = models.DateTimeField(auto_now=True, verbose_name="更新时间")

    class Meta:
        verbose_name = "对话会话"
        verbose_name_plural = "对话会话"
        ordering = ['-updated_at']

    def __str__(self):
        return self.title


class Message(models.Model):
    """消息模型"""
    ROLE_CHOICES = [
        ('user', '用户'),
        ('assistant', '助手'),
    ]

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name='messages',
        verbose_name="所属会话"
    )
    role = models.CharField(max_length=10, choices=ROLE_CHOICES, verbose_name="角色")
    content = models.TextField(verbose_name="消息内容")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
    
    # 存储用户原始问题（意图识别前的）
    original_question = models.TextField(blank=True, null=True, verbose_name="原始问题")

    class Meta:
        verbose_name = "消息"
        verbose_name_plural = "消息"
        ordering = ['created_at']

    def __str__(self):
        return f"{self.role}: {self.content[:50]}"
