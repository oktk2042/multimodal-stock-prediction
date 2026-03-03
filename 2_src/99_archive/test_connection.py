import os
from openai import AzureOpenAI
from dotenv import load_dotenv

# .envファイルを読み込む
load_dotenv()

def test_gpt():
    print("--- Azure OpenAI 接続テスト開始 ---")

    # 1. 環境変数の読み込み
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    print(f"ターゲットデプロイ: {deployment}")
    print(f"エンドポイント: {endpoint}")

    # 2. クライアントの初期化
    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version
    )

    # 3. テストリクエスト
    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": "あなたはシステム管理者です。",
            },
            {
                "role": "user",
                "content": "接続テスト成功です、とだけ返してください。",
            }
        ],
        max_completion_tokens=16384,
    )
    
    print("\n✅ 成功しました！")
    print("-" * 30)
    print("AIからの応答: " + response.choices[0].message.content)
    print("-" * 30)

if __name__ == "__main__":
    test_gpt()