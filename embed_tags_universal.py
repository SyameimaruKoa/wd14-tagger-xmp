import http.server
import socketserver
import os
import io
import json
import subprocess
import sys
import shutil
import datetime
import time

# 必要なライブラリのインポートチェック
try:
    from PIL import Image
    import numpy as np
    import onnxruntime as ort
    import pandas as pd
except ImportError as e:
    print(f"必要なライブラリが足りぬようじゃ: {e}")
    print("pip install pillow numpy onnxruntime pandas")
    sys.exit(1)

# ---------------------------------------------------------
# 設定・定数
# ---------------------------------------------------------
PORT = 5000
# モデルのパス（適宜書き換えるのじゃ）
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")
TAGS_PATH = os.path.join(MODEL_DIR, "selected_tags.csv")

# ---------------------------------------------------------
# ExifTool 自動インストール関数
# ---------------------------------------------------------
def check_and_install_exiftool():
    """ExifToolがシステムにあるか確認し、なければUbuntu(apt)でインストールを試みる"""
    if shutil.which('exiftool'):
        return

    print("\n[System] ExifToolが見当たらぬ。インストールを試みるぞ...")
    try:
        # root権限でなければsudoをつける
        cmd_prefix = [] if os.geteuid() == 0 else ['sudo']
        
        # apt-get update
        print("[System] パッケージリストを更新中...")
        subprocess.run(cmd_prefix + ['apt-get', 'update'], check=True)
        
        # apt-get install
        print("[System] libimage-exiftool-perl をインストール中...")
        subprocess.run(cmd_prefix + ['apt-get', 'install', '-y', 'libimage-exiftool-perl'], check=True)
        
        print("[System] インストール完了じゃ。\n")
    except subprocess.CalledProcessError as e:
        print(f"[Error] インストールに失敗したようじゃ... 手動で入れてくれ。\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[Error] 予期せぬエラーじゃ: {e}")
        sys.exit(1)

# ---------------------------------------------------------
# 推論エンジンのロード (前と同じ)
# ---------------------------------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(TAGS_PATH):
    print(f"モデルファイルが見つからぬ！ {MODEL_DIR} を確認せよ。")
    sys.exit(1)

print("[System] Loading model...")
# GPUが使えるならGPUで、だめならCPU
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    ort_sess = ort.InferenceSession(MODEL_PATH, providers=providers)
except Exception:
    ort_sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])

input_name = ort_sess.get_inputs()[0].name
df_tags = pd.read_csv(TAGS_PATH)

def preprocess_image(image):
    # 画像の前処理
    image = image.convert("RGB")
    image = image.resize((448, 448), Image.BICUBIC)
    image_np = np.array(image).astype(np.float32)
    # BGR -> RGB (PILはRGBだがモデルによっては並び順注意。WD14はBGR推奨の場合が多いが、
    # 多くの実装でRGBのまま突っ込んでも動く。ここでは一般的なBGR変換を入れる)
    image_np = image_np[:, :, ::-1] 
    image_np = np.expand_dims(image_np, 0)
    return image_np

def infer_tags(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        input_data = preprocess_image(image)
        
        probs = ort_sess.run(None, {input_name: input_data})[0][0]
        
        # 閾値0.35以上のタグを抽出
        tag_text = ""
        for i, p in enumerate(probs):
            if p >= 0.35 and i < len(df_tags):
                tag_name = df_tags.iloc[i]['name']
                tag_text += tag_name + ", "
        
        return tag_text.strip(", ")
    except Exception as e:
        print(f"[Error] Inference failed: {e}")
        return ""

# ---------------------------------------------------------
# サーバーハンドラ
# ---------------------------------------------------------
class TaggingHandler(http.server.BaseHTTPRequestHandler):
    
    # デフォルトのログ出力を抑制・整形する
    def log_message(self, format, *args):
        # ステータスコードが200の場合は標準ログを出さない（do_POSTで自前で出す）
        if args[1] == '200':
            return
        # エラーや404などは表示する
        sys.stderr.write("%s - - [%s] %s\n" %
                         (self.client_address[0],
                          self.log_date_time_string(),
                          format % args))

    def do_POST(self):
        try:
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)

            # 推論実行
            tags = infer_tags(post_data)
            
            # レスポンス
            self.send_response(200)
            self.send_header('Content-type', 'text/plain; charset=utf-8')
            self.end_headers()
            self.wfile.write(tags.encode('utf-8'))

            # --- 見やすいログ出力 ---
            # サイズ計算
            size_kb = len(post_data) / 1024
            size_str = f"{size_kb:.1f} KB"
            if size_kb > 1024:
                size_str = f"{size_kb/1024:.2f} MB"

            # タイムスタンプ
            now = datetime.datetime.now().strftime("%H:%M:%S")
            
            # ログ表示 [時刻] IP | Size | Status
            print(f"[{now}] {self.client_address[0]:<15} | Size: {size_str:<9} | Processed OK")

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            print(f"[Error] Request handling failed: {e}")

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
if __name__ == "__main__":
    # まずExifToolをチェック
    check_and_install_exiftool()

    # サーバーIP取得（LAN内表示用）
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()

    print("\n" + "="*40)
    print(f" [INFO] Server running on Port {PORT}")
    print(f" [INFO] LAN IP: {IP}")
    print(f" [INFO] Ready. Logs will be concise.")
    print("="*40 + "\n")

    with socketserver.TCPServer(("", PORT), TaggingHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[System] Stopping server...")
            httpd.server_close()