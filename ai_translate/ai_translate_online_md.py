import argparse
import logging
import os
import sys

import openai
import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

system_prompt_template = """
# 角色和目标

你是一位精通中英双语的专业级技术翻译专家，拥有超过十年的软件工程和计算机科学领域文档翻译经验。你的目标是将我提供的英文技术文档翻译成一篇专业、精准、流畅且符合中国开发者阅读习惯的简体中文技术文章。

# 翻译准则

请在翻译过程中严格遵守以下准则：

1.  **忠实原文 (Fidelity):**
    *   翻译必须在信息层面完全忠实于原文，不得添加、删减或曲解任何技术细节和核心思想。
    *   精准传达原文的细微差别和意图。

2.  **专业术语 (Terminology):**
    *   对于技术术语（如框架、协议、算法、设计模式等），请遵循以下优先级处理：
        1.  **官方/行业标准译名:** 优先使用已存在的、广为接受的官方译名或行业标准译名。
        2.  **约定俗成译名:** 若无官方译名，则使用中文技术社区约定俗成的通用译名。
        3.  **保留英文原文:** 如果一个术语在中文技术圈中普遍以英文形式出现（例如：`API`, `CPU`, `Docker`, `Git`, `Hook`, `CI/CD`），请直接保留英文原文，以避免不必要的翻译造成混淆。
        4.  **首次出现时注解:** 对于一个可能不那么广为人知但保留英文更合适的术语，可以在其首次出现时，在括号内提供简短的中文注解。格式为：`英文术语 (中文注解)`。
        5.  **无对应译名:** 如果遇到没有合适中文对应的全新术语，请在审慎翻译后，在括号内附上英文原文。格式为：`新译名 (Original English Term)`。

3.  **流畅自然 (Fluency & Naturalness):**
    *   译文必须通顺流畅，符合现代简体中文的语法和表达习惯，杜绝"翻译腔"。
    *   在保证技术准确性的前提下，灵活调整句式结构，使其更易于理解和阅读。例如，可以将英文的长句拆分为更符合中文阅读习惯的短句。

4.  **格式保留 (Formatting):**
    *   完整保留原文的所有格式，包括但不限于 Markdown 标记、代码块、超链接、段落结构等。

5.  **语气风格 (Tone & Style):**
    *   保持与原文一致的专业、客观、严谨的行文风格。

# 操作指令

1.  请仔细阅读并深刻理解我提供的整篇英文文档的上下文。
2.  请严格遵循上述所有【翻译准则】进行翻译。
3.  直接输出翻译后的简体中文内容，不要包含任何额外的说明、开场白或结束语。
"""


class MarkdownTranslator:
    """Markdown翻译器 - 专注于翻译功能"""
    
    def __init__(self, api_key, api_base=None, model="deepseek-chat"):
        """
        初始化翻译器
        
        Args:
            api_key (str): OpenAI API密钥
            api_base (str, optional): API基础URL
            model (str): 使用的模型名称
        """
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        
        # 配置OpenAI客户端
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base
        )
    
    def translate(self, markdown_content):
        """
        翻译Markdown内容
        
        Args:
            markdown_content (str): 要翻译的Markdown内容
            
        Returns:
            str: 翻译后的中文Markdown内容
            
        Raises:
            ValueError: 当输入内容为空时
            Exception: 翻译过程中的其他错误
        """
        if not markdown_content or not markdown_content.strip():
            logger.warning("输入的Markdown内容为空")
            raise ValueError("输入的Markdown内容不能为空")
        
        try:
            logger.info("开始翻译Markdown内容")
            
            # 构建用户消息
            user_message = f"请翻译以下英文技术文档：\n\n{markdown_content}"
            
            # 调用AI接口进行翻译
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt_template},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,  # 较低的温度以确保翻译的准确性
                max_tokens=64000   # 根据模型能力调整
            )
            
            translated_content = response.choices[0].message.content
            logger.info("翻译完成")
            
            return translated_content
            
        except Exception as e:
            logger.error(f"翻译过程中出错: {str(e)}")
            raise


def fetch_markdown_from_url(url, timeout=30):
    """
    从URL获取Markdown内容
    
    Args:
        url (str): 目标URL
        timeout (int): 请求超时时间（秒）
        
    Returns:
        str: 获取到的Markdown内容
        
    Raises:
        requests.RequestException: 请求失败时抛出
    """
    try:
        logger.info(f"正在从URL获取Markdown内容: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        content = response.text
        logger.info(f"成功获取内容，长度: {len(content)} 字符")
        return content
        
    except requests.RequestException as e:
        logger.error(f"获取Markdown内容失败: {str(e)}")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="从URL获取Markdown内容并翻译成中文",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python ai_translate_online_md.py --url "https://example.com/doc.md" --output translated.md
  python ai_translate_online_md.py --url "https://example.com/doc.md" --api-key "your-api-key"
        """
    )
    
    parser.add_argument("--url",required=True,help="包含Markdown内容的URL")
    parser.add_argument("--api-key",required=True,help="OpenAI API密钥")
    parser.add_argument("--api-base",help="OpenAI API基础URL（可选）")
    parser.add_argument("--model",default="deepseek-chat",help="使用的AI模型（默认: deepseek-chat）")
    parser.add_argument("--output",help="输出文件路径（可选，不指定则只打印到控制台）")
    parser.add_argument("--timeout",type=int,default=30,help="HTTP请求超时时间（秒，默认: 30）")
    
    args = parser.parse_args()
    
    try:
        # 从URL获取Markdown内容
        markdown_content = fetch_markdown_from_url(args.url, args.timeout)
        
        # 创建翻译器实例
        translator = MarkdownTranslator(
            api_key=args.api_key,
            api_base=args.api_base,
            model=args.model
        )
        
        # 执行翻译
        translated_content = translator.translate(markdown_content)
        
        # 保存到文件（如果指定）
        if args.output:
            try:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(translated_content)
                logger.info(f"翻译结果已保存到: {os.path.abspath(args.output)}")
            except Exception as e:
                logger.error(f"保存文件失败: {str(e)}")
                raise
        
        # 如果没有指定输出文件，则打印到控制台
        if not args.output:
            print("\n" + "="*50)
            print("翻译结果:")
            print("="*50)
            print(translated_content)
        
        logger.info("翻译任务完成")
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
        sys.exit(1)
    except Exception as e:
        logger.error(f"程序执行失败: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()