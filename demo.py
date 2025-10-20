"""
簡易示範腳本：展示如何使用運動分類系統

使用方式：
1. 安裝套件: pip install -r requirements.txt
2. 執行範例: python demo.py

版本說明：
- inference.py: 基礎版（快速測試用）
- inference_v2.py: 完整版（符合立偉模型的實作）
"""

from inference_v2 import ExerciseClassifierV2
import os

def demo_single_video():
    """單一影片分類範例"""
    print("=== 單一影片分類範例 ===\n")

    # 初始化分類器（使用完整版）
    classifier = ExerciseClassifierV2(model_path="bilstm_mix_best_pt.pth")

    # 讓使用者輸入影片路徑
    video_path = input("請輸入影片檔名或路徑（例如: squat.mp4）: ").strip()

    if not video_path:
        print("未輸入檔名，使用預設值: example_video.mp4")
        video_path = "example_video.mp4"

    if not os.path.exists(video_path):
        print(f"錯誤: 找不到影片檔案 '{video_path}'")
        print("請確認檔案名稱是否正確，或將影片放在專案目錄下")
        return

    # 進行預測（不儲存輸出影片）
    try:
        predicted_class, confidence, probabilities = classifier.predict_video(video_path)

        print(f"\n預測結果: {classifier.class_names[predicted_class]}")
        print(f"信心度: {confidence:.2%}")

        print("\n所有類別機率:")
        for i, (name, prob) in enumerate(zip(classifier.class_names, probabilities)):
            marker = " ← 預測" if i == predicted_class else ""
            print(f"  {name}: {prob:.2%}{marker}")
    except Exception as e:
        print(f"處理影片時發生錯誤: {e}")

def demo_with_visualization():
    """帶視覺化輸出的範例"""
    print("=== 帶視覺化的分類範例 ===\n")

    classifier = ExerciseClassifierV2()

    # 讓使用者輸入影片路徑
    video_path = input("請輸入影片檔名或路徑（例如: squat.mp4）: ").strip()

    if not video_path:
        print("未輸入檔名，使用預設值: example_video.mp4")
        video_path = "example_video.mp4"

    if not os.path.exists(video_path):
        print(f"錯誤: 找不到影片檔案 '{video_path}'")
        print("請確認檔案名稱是否正確，或將影片放在專案目錄下")
        return

    # 輸出檔名
    output_path = input("請輸入輸出影片檔名（直接按 Enter 使用預設: output_with_skeleton.mp4）: ").strip()
    if not output_path:
        output_path = "output_with_skeleton.mp4"

    # 進行預測並產生視覺化影片
    try:
        classifier.predict_and_visualize(video_path, output_path)
    except Exception as e:
        print(f"處理影片時發生錯誤: {e}")

def demo_batch_processing():
    """批次處理多個影片的範例"""
    print("=== 批次處理範例 ===\n")

    classifier = ExerciseClassifierV2()

    # 讓使用者輸入影片清單
    print("請輸入要處理的影片檔名（每行一個，輸入空行結束）:")
    video_files = []
    while True:
        filename = input(f"影片 {len(video_files)+1} (直接按 Enter 結束): ").strip()
        if not filename:
            break
        video_files.append(filename)

    if not video_files:
        print("未輸入任何影片，使用範例清單")
        video_files = ["squat1.mp4", "benchpress1.mp4", "deadlift1.mp4"]

    results = []
    for video_file in video_files:
        if not os.path.exists(video_file):
            print(f"跳過: {video_file} (檔案不存在)")
            continue

        try:
            print(f"\n正在處理: {video_file}...")
            predicted_class, confidence, _ = classifier.predict_video(video_file)
            results.append({
                'file': video_file,
                'prediction': classifier.class_names[predicted_class],
                'confidence': confidence
            })
        except Exception as e:
            print(f"處理 {video_file} 時發生錯誤: {e}")

    # 顯示結果摘要
    print("\n=== 批次處理結果 ===")
    if results:
        for result in results:
            print(f"{result['file']}: {result['prediction']} ({result['confidence']:.2%})")
    else:
        print("沒有成功處理任何影片")

if __name__ == "__main__":
    print("運動動作分類系統示範\n")
    print("請選擇示範模式:")
    print("1. 單一影片分類")
    print("2. 帶視覺化輸出")
    print("3. 批次處理")

    choice = input("\n請輸入選項 (1-3): ").strip()

    if choice == "1":
        demo_single_video()
    elif choice == "2":
        demo_with_visualization()
    elif choice == "3":
        demo_batch_processing()
    else:
        print("無效的選項，執行預設示範...")
        demo_single_video()
