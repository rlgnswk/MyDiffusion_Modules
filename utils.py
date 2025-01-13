import numpy as np
from PIL import Image

def save_concatenated_image(
    image_list, 
    save_path="./output/combined_result.png", 
    orientation="horizontal"
):
    """
    image_list: NumPy 배열 형태의 이미지들을 담은 리스트
    save_path: 합쳐진 이미지를 저장할 경로
    orientation: 'horizontal' 또는 'vertical'
    """
    # 만약 이미지 크기가 다르면, 리사이즈 과정을 거쳐야 함 (생략 가능)
    # 여기서는 이미지 크기가 모두 동일하다고 가정
    
    # 이미지 병합
    if orientation == "horizontal":
        # 수평으로 이어붙이기
        merged = np.concatenate(image_list, axis=1)
    else:
        # 수직으로 이어붙이기
        merged = np.concatenate(image_list, axis=0)
    
    # NumPy 배열 -> Pillow Image 변환
    merged_pil = Image.fromarray(merged)
    
    # 저장
    merged_pil.save(save_path)
    print(f"Concatenated image saved at: {save_path}")