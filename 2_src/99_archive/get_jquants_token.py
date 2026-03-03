import requests
import json
import getpass

def get_refresh_token():
    print("--- J-Quants Refresh Token 取得ツール ---")
    email = input("J-Quants登録メールアドレス: ")
    password = getpass.getpass("J-Quantsパスワード (入力は見えません): ")

    url = "https://api.jquants.com/v1/token/auth_user"
    headers = {"Content-Type": "application/json"}
    data = {
        "mailaddress": email,
        "password": password
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        token = response.json().get("refreshToken")
        print("\n✅ 取得成功！以下のトークンを .env ファイルの JQUANTS_REFRESH_TOKEN に貼り付けてください:\n")
        print(token)
        print("\n--------------------------------------------------")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        if response.status_code == 400 or response.status_code == 401:
            print("メールアドレスかパスワードが間違っている可能性があります。")

if __name__ == "__main__":
    get_refresh_token()