"""一次性工具：生成 ORB 测试素材。不是练习内容，用完可删。

生成 4 张图：
    a_original.jpg / a_rotated.jpg   —— 只有旋转不同
    b_original.jpg / b_scaled.jpg    —— 只有尺度不同
"""
import cv2
import numpy as np

W, H = 1200, 900
OUT = "images"
QUALITY = 95


def make_base(seed):
    """画一张纹理丰富的合成图：渐变底 + 噪点 + 多边形 + 圆 + 文字 + 棋盘格。"""
    rng = np.random.default_rng(seed)

    # 渐变底 + 噪点（保证没有大片纯色）
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = xx / W * 90 + yy / H * 90
    img = np.dstack([base, base * 0.85, base * 0.7])
    img = np.clip(img + rng.integers(-25, 25, (H, W, 3)), 0, 255).astype(np.uint8)

    # 随机多边形
    for _ in range(45):
        pts = rng.integers(0, [W, H], size=(int(rng.integers(3, 7)), 2)).astype(np.int32)
        color = tuple(int(c) for c in rng.integers(40, 255, 3))
        cv2.fillPoly(img, [pts], color)

    # 随机圆
    for _ in range(40):
        cx, cy = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        color = tuple(int(c) for c in rng.integers(40, 255, 3))
        cv2.circle(img, (cx, cy), int(rng.integers(12, 70)), color, -1)

    # 棋盘格小块（规则纹理，角点密集）
    for _ in range(6):
        x0, y0 = int(rng.integers(0, W - 200)), int(rng.integers(0, H - 200))
        cell = int(rng.integers(10, 22))
        for i in range(0, 180, cell):
            for j in range(0, 180, cell):
                if (i // cell + j // cell) % 2 == 0:
                    cv2.rectangle(img, (x0 + i, y0 + j), (x0 + i + cell, y0 + j + cell),
                                  (240, 240, 240), -1)

    # 文字（字母边缘提供大量角点）
    words = ["ORB", "FEATURE", "MATCH", "KEYPOINT", "DESCRIPTOR",
             "HAMMING", "BFMatcher", "PYTHON", "OPENCV", "2026"]
    for _ in range(28):
        s = words[int(rng.integers(0, len(words)))]
        org = (int(rng.integers(0, W - 260)), int(rng.integers(30, H - 20)))
        scale = float(rng.uniform(0.7, 2.4))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                    int(rng.integers(2, 5)), cv2.LINE_AA)

    # 随机线段
    for _ in range(60):
        p1 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        p2 = (int(rng.integers(0, W)), int(rng.integers(0, H)))
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.line(img, p1, p2, color, int(rng.integers(2, 6)), cv2.LINE_AA)

    return img


def rotate_expand(img, angle):
    """旋转并扩大画布，保证四个角不被裁掉。"""
    h, w = img.shape[:2]
    diag = int(np.ceil(np.hypot(h, w)))
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    m[0, 2] += (diag - w) / 2.0
    m[1, 2] += (diag - h) / 2.0
    return cv2.warpAffine(img, m, (diag, diag), borderMode=cv2.BORDER_REPLICATE)


def main():
    a = make_base(20260918)
    b = make_base(777)

    cv2.imwrite(f"{OUT}/a_original.jpg", a, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    cv2.imwrite(f"{OUT}/a_rotated.jpg", rotate_expand(a, 40.0),
                [cv2.IMWRITE_JPEG_QUALITY, QUALITY])

    cv2.imwrite(f"{OUT}/b_original.jpg", b, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
    small = cv2.resize(b, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUT}/b_scaled.jpg", small, [cv2.IMWRITE_JPEG_QUALITY, QUALITY])

    for name in ["a_original", "a_rotated", "b_original", "b_scaled"]:
        im = cv2.imread(f"{OUT}/{name}.jpg")
        print(f"{name:12s} {str(im.shape):18s} 文件OK")


if __name__ == "__main__":
    main()
