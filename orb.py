import cv2
import sys
from pathlib import Path
import numpy as np

TOP_N = 60    
N_FEATURES = 500
BASE = Path(__file__).parent

PAIRS = [
    ("旋转", "rotate", "a_original.jpg", "a_rotated.jpg"),
    ("尺度", "scale", "b_original.jpg", "b_scaled.jpg"),
    ("无关", "unrel", "a_original.jpg", "b_original.jpg"),
]

def load_image(name):
    path = BASE / "images" / name
    rgb = cv2.imread(path)

    if rgb is None:
        return None
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return {"rgb": rgb, "gray": gray}

def detect(gray, orb):
    kp, des = orb.detectAndCompute(gray, None)
    if des is None:
        print("计算失败，无足够的关键点")
        return None, None
    return kp, des

def match_bfm(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return matches

def sort_matches(matches, n):
    matches_sorted = sorted(matches, key = lambda m:m.distance)
    matches_topn = matches_sorted[:n]
    return matches_topn

def run_pair(name, img1_name, img2_name, orb, pid):

    img1 = load_image(img1_name)
    img2 = load_image(img2_name)
    if img1 is None or img2 is None:
        print("读图失败")
        return None

    #特征提取
    kp1, des1 = detect(img1["gray"], orb)
    kp2, des2 = detect(img2["gray"], orb)
    if des1 is None or des2 is None:
        print("无提取到的特征")
        return None
    #bfm匹配 第一步过纹理
    matches = match_bfm(des1, des2)
    matches_topn = sort_matches(matches, TOP_N)
    #使用ransac匹配最少有四个相似的几何关系
    if len(matches) < 4:
        print(name, "匹配数不足 4,无法求解几何关系")
        return
    #ransac匹配 第二步过位置
    L1 = [kp1[m.queryIdx].pt for m in matches]
    L2 = [kp2[m.trainIdx].pt for m in matches]
    src = np.array(L1, dtype = np.float32)
    dst = np.array(L2, dtype = np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3)
    #点数匹配需求满足仍可能退化
    if H is None or mask is None:
        print("无满足的几何关系")
        return
    inliers = [m for m, k in zip(matches, mask.ravel()) if k == 1]
    print(name, "匹配数量", len(matches), "最佳匹配的汉明距离", matches_topn[0].distance, "所有匹配的中位距离", np.median([m.distance for m in matches]))
    print("内点数量-通过ransac匹配", len(inliers))
    #计算内点率，设置噪声地板
    inlier_rate = len(inliers) / len(matches)
    print("内点率为：", inlier_rate)
    draw(img1["rgb"], kp1, img2["rgb"], kp2, matches_topn, 0, f"{pid}:bfm")
    if inlier_rate > 0.15:
        draw(img1["rgb"], kp1, img2["rgb"], kp2, inliers, 0, f"{pid}:bfm+ransac")
    

  
def draw(img1, kp1, img2, kp2, matches_to_draw, wait_ms, title):
    answer = cv2.drawMatches(img1, kp1, img2, kp2, matches_to_draw, None, cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, answer)
    cv2.waitKey(wait_ms)
    cv2.destroyAllWindows()


def main():

    orb = cv2.ORB_create(nfeatures = N_FEATURES)
    which = sys.argv[1] if len(sys.argv) > 1 else None
    for name, pid, f1, f2 in PAIRS:
        if  which is not None and pid != which:
            continue
        run_pair(name, f1, f2, orb, pid)



    


if __name__ == "__main__":
    main()

