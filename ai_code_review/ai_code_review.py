import argparse
import difflib
import logging
import os
import sys

import openai
from git import Repo
from git.exc import GitError

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)


class GitDiffReviewer:
    def __init__(self, repo_path, base_branch, feature_branch, api_key, api_base=None, model="deepseek-chat"):
        self.repo_path = repo_path
        self.base_branch = base_branch
        self.feature_branch = feature_branch
        self.api_key = api_key
        self.api_base = api_base
        self.model = model

        try:
            self.repo = Repo(self.repo_path)
        except GitError as e:
            logger.error(f"无法打开 Git 仓库: {str(e)}")
            raise

    def get_diff(self, output_file=None):
        """
        获取 feature 分支相对于 base 分支的增量修改
        使用 merge_base 确保只获取 feature 分支独有的变更
        """
        try:
            # 获取 merge base（分叉点）
            merge_bases = self.repo.merge_base(self.base_branch, self.feature_branch)
            if not merge_bases:
                logger.warning(f"分支 {self.base_branch} 和 {self.feature_branch} 没有共同祖先")
                return None

            fork_point = merge_bases[0]  # 取第一个 merge base
            feature_head = self.repo.commit(self.feature_branch)

            # 获取差异（分叉点..feature_head）
            diff_index = fork_point.diff(feature_head,
                                         create_patch=True,
                                         ignore_blank_lines=True,
                                         ignore_space_at_eol=True)

            # 支持的编码列表（优先尝试 UTF-8，最后回退到 latin1）
            encodings = ['utf-8', 'gbk']

            diff_content = ""
            processed_files = set()

            for diff_item in diff_index:
                try:
                    file_path = diff_item.b_path if diff_item.b_path else diff_item.a_path

                    # 跳过已处理文件（避免重复）
                    if file_path in processed_files:
                        continue
                    processed_files.add(file_path)

                    # 处理不同类型的变更
                    if diff_item.new_file:  # 新增文件
                        blob = diff_item.b_blob
                        content = self._decode_blob(blob, encodings)
                        if content is not None:
                            diff_content += f"新文件: {file_path}\n{content}\n\n"

                    elif diff_item.deleted_file:  # 删除文件
                        diff_content += f"删除的文件: {file_path}\n\n"

                    else:  # 修改的文件
                        a_blob = diff_item.a_blob
                        b_blob = diff_item.b_blob

                        old_content = self._decode_blob(a_blob, encodings)
                        new_content = self._decode_blob(b_blob, encodings)

                        if old_content is not None and new_content is not None:
                            diff_lines = difflib.unified_diff(
                                old_content.splitlines(keepends=True),
                                new_content.splitlines(keepends=True),
                                fromfile=f"a/{file_path}",
                                tofile=f"b/{file_path}",
                                lineterm=''
                            )
                            diff_content += f"修改文件: {file_path}\n"
                            diff_content += "".join(diff_lines) + "\n\n"

                except Exception as e:
                    logger.error(f"处理文件 {file_path} 时出错: {str(e)}", exc_info=True)
                    continue

            # 输出到文件（如果指定）
            if output_file and diff_content.strip():
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(diff_content)
                logger.info(f"差异已保存到: {os.path.abspath(output_file)}")

            return diff_content if diff_content.strip() else None

        except Exception as e:
            logger.error("获取增量差异时出错", exc_info=True)
            raise

    def _decode_blob(self, blob, encodings):
        """尝试用多种编码解码 blob 内容"""
        if blob is None:
            return None

        raw_data = blob.data_stream.read()
        for encoding in encodings:
            try:
                return raw_data.decode(encoding)
            except UnicodeDecodeError:
                continue
        logger.warning(f"无法解码文件 {blob.path}，尝试使用 latin1")
        return raw_data.decode('latin1', errors='replace')  # 最终回退

    def review_code(self, diff_content):
        """使用AI接口审查代码"""
        if not diff_content:
            logger.info("没有发现代码差异")
            return

        logger.info("正在审查代码...")

        try:
            system_prompt = (
                    "你是一个专业的开发人员，请审查以下代码差异，指出潜在问题并提供改进建议。"
                    "请按照以下格式反馈：\n"
                    "1. 文件路径\n"
                    "2. 问题行号\n"
                    "3. 问题描述\n"
                    "4. 改进建议\n\n"
            )
            user_prompt = (
                    "代码差异:\n" + diff_content
            )

            client_config = {"api_key": self.api_key, "timeout": 90}
            if self.api_base:
                client_config["base_url"] = self.api_base

            client = openai.OpenAI(**client_config)
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )

            response_message = response.choices[0].message
            reason_result = getattr(response_message, "reasoning_content", None)
            if reason_result:
                print(f"<think>\n{reason_result}\n</think>")
            review_result = response.choices[0].message.content
            print("审查结果：")
            print(review_result)

        except Exception as e:
            logger.error("审查过程中出错", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(description='Git 代码差异审查工具')
    parser.add_argument('--repo', required=True, help='Git 仓库路径')
    parser.add_argument('--base', required=True, help='基础分支名称')
    parser.add_argument('--feature', required=True, help='特性分支名称')
    parser.add_argument('--api-key', required=True, help='API 密钥')
    parser.add_argument('--api-base', help='自定义 API 基础地址')
    parser.add_argument('--model', default='deepseek-chat', help='模型名称')
    parser.add_argument('--output-diff', help='将差异输出到指定文件（用于验证git diff结果是否正确）')
    args = parser.parse_args()

    try:
        reviewer = GitDiffReviewer(
            repo_path=args.repo,
            base_branch=args.base,
            feature_branch=args.feature,
            api_key=args.api_key,
            api_base=args.api_base,
            model=args.model
        )

        # 获取差异并输出到文件
        diff_content = reviewer.get_diff(output_file=args.output_diff)

        # 如果只想输出差异文件，不进行AI审查，可以添加以下判断
        if args.output_diff and not diff_content:
            logger.warning("没有发现差异内容，未生成输出文件")
            return

        reviewer.review_code(diff_content)

    except Exception as e:
        logger.critical("程序运行失败", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()