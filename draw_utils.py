
import openslide
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from typing import Dict
from collections import Counter
from openslide.deepzoom import DeepZoomGenerator

import os, math
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from xml.dom import minidom

COLOR_MAP = {
    "Apoptotic Body":    "#6F4E37",  # 고동색 (짙은 갈색/마룬 톤)
    "Cancer cell":       "#8B0000",  # 검붉은색 (다크 레드)
    "Endothelial Cell":  "#87CEEB",  # 하늘색 (스카이 블루)
    "Eosinophils":       "#ADFF2F",  # 연두색 (그린옐로)
    "Epithelial":        "#FF7F00",  # 주황색 (비비드 오렌지)
    "Fibroblasts":       "#1F3A93",  # 파란색 (진한 로열 블루)
    "Lymphocytes":       "#00BFCF",  # 시안 (청록/사이언)
    "Macrophages":       "#4DAF4A",  # 초록색 (비비드 그린)
    "Minor Stromal Cell":"#C49A6C",  # 옅은 갈색 (라이트 브라운)
    "Mitotic Figures":   "#4B0082",  # 짙은 보라색 (인디고 퍼플)
    "Muscle Cell":       "#003366",  # 남색 (딥 네이비)
    "Neutrophils":       "#FFD700",  # 노란색 (골드/비비드 옐로)
    "Plasmocytes":       "#B19CD9",  # 연보라색 (라벤더)
    "Red blood cell":    "#FF69B4",  # 핫핑크 (핫 핑크)
}
# 혹시 목록에 없는 문자열이 들어오면 쓸 기본색
DEFAULT_COLOR = "#ffffff"

def normalize_cell_type(name: str) -> str:
    if not name:
        return "unknown"
    n = name.strip().lower()
    # 복수형/표현 차이 보정
    if n in ["fibroblast", "fibroblasts"]:
        return "fibroblast"
    if n in ["cancer", "cancer cell", "cancer cells"]:
        return "cancer"
    if n in ["lymphocyte", "lymphocytes"]:
        return "lymphocyte"
    if n in ["macrophage", "macrophages"]:
        return "macrophage"
    if n in ["plasmocyte", "plasmocytes"]:
        return "plasmocyte"
    if n in ["endothelial cell", "endothelial cells"]:
        return "endothelial"
    if n in ["red blood cell", "red blood cells", "rbc"]:
        return "rbc"
    # 그 외 그대로 반환
    return n


def get_bounds_offset(slide: openslide.OpenSlide):
    bx = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_X, 0) or 0)
    by = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_Y, 0) or 0)
    bw = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_WIDTH, 0) or 0)
    bh = int(slide.properties.get(openslide.PROPERTY_NAME_BOUNDS_HEIGHT, 0) or 0)
    return bx, by, bw, bh

def dz_pixel_to_base(slide, dz, dz_level, x_dz, y_dz):
    W0, H0 = slide.dimensions
    W_dz, H_dz = dz.level_dimensions[dz_level]
    sx = W0 / float(W_dz)
    sy = H0 / float(H_dz)
    bx, by, _, _ = get_bounds_offset(slide)
    x0 = int(round(x_dz * sx)) + bx
    y0 = int(round(y_dz * sy)) + by
    return x0, y0

def dz_tile_to_base(slide, dz, dz_level, tx, ty, tile_size=224, overlap=0):
    step = tile_size - overlap
    x_dz = tx * step
    y_dz = ty * step
    return dz_pixel_to_base(slide, dz, dz_level, x_dz, y_dz)

def dz_tile_base_size(slide, dz, dz_level, tile_size=224, overlap=0):
    step = tile_size - overlap
    W0, H0 = slide.dimensions
    W_dz, H_dz = dz.level_dimensions[dz_level]
    sx = W0 / float(W_dz)
    sy = H0 / float(H_dz)
    w0 = int(round(step * sx))
    h0 = int(round(step * sy))
    return w0, h0

def get_base_patch(tile, slide):
        
    # Histoplus가 쓰는 tile_size=224, overlap=0 가정
    dz = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=True)
    lvl = int(tile['level'])
    tx, ty = int(tile['x']), int(tile['y'])

    # 1) 타일 좌상단의 base 좌표
    x0, y0 = dz_tile_to_base(slide, dz, lvl, tx, ty, tile_size=224, overlap=0)

    # 2) 한 칸의 base 폭/높이 (모서리 타일은 실제 더 작을 수도 있음)
    w0, h0 = dz_tile_base_size(slide, dz, lvl, tile_size=224, overlap=0)
    
    # print(f"해당 타일에서 관찰된 cell types → {', '.join(tile['types'])}")

    tile_cfg = {
        'lvl': lvl,
        'txy': (tx,ty),
        'xy0': (x0,y0),
        'wh0': (w0,h0)
    }
    # 3) read_region으로 “원본(level-0)”에서 정확한 윈도우 추출
    region_base = slide.read_region((x0, y0), 0, (w0, h0)).convert("RGB")
    return region_base, tile_cfg

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def load_font(size=14):
    # 가능한 폰트 로드 (환경에 따라 경로 조정)
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except:
        return ImageFont.load_default()

def draw_masks(dz, tile_masks, region_base, TILE_INFO:Dict, COLOR_MAP = COLOR_MAP, DEFAULT_COLOR = DEFAULT_COLOR):
    '''
    W_dz, H_dz = dz.level_dimensions[lvl]
    sx, sy = W0/float(W_dz), H0/float(H_dz)   # dz 픽셀 → base 픽셀 스케일

    # 타일 좌상단 base 좌표 (overlap=0 → step=tile_size)
    step = int(tile["width"])                  # 보통 224
    x0 = int(round(tx * step * sx))
    y0 = int(round(ty * step * sy))

    # 실제 DZ 타일 크기(가장자리 보정)
    img_dz = dz.get_tile(lvl, (tx, ty))        # PIL, dz level의 실제 타일 크기
    w0 = int(round(img_dz.width  * sx))
    h0 = int(round(img_dz.height * sy))
    '''

    # Histoplus가 쓰는 tile_size=224, overlap=0 가정
    
    lvl = int(TILE_INFO['lvl'])
    tx, ty = int(TILE_INFO['txy'][0]), int(TILE_INFO['txy'][1])

    img_dz = dz.get_tile(lvl, (tx, ty))        # PIL, dz level의 실제 타일 크기

    # --- 폴리곤 그릴 때 사용할 스케일: (타일 로컬 → base region 크기) ---
    scale_x = int(TILE_INFO['wh0'][0]) / float(img_dz.width)         # 예: 448/224 = 2.0
    scale_y = int(TILE_INFO['wh0'][0]) / float(img_dz.height)

    region = region_base.copy()
    draw = ImageDraw.Draw(region, "RGBA")
    # 이 tile에 속한 mask들만 추출

    # --- 폴리곤/센트로이드 오버레이 ---
    for m in tile_masks:
        ctype = m.get("cell_type")
        color = COLOR_MAP.get(ctype, DEFAULT_COLOR)

        # 타일 로컬(≈224×224) 좌표 → base region 좌표로 스케일링
        pts = [(x*scale_x, y*scale_y) for (x, y) in m["coordinates"]]
        draw.line(pts + [pts[0]], fill=color, width=2)

        if "centroid" in m:
            cx, cy = m["centroid"]
            cx, cy = cx*scale_x, cy*scale_y
            r = 3
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], outline="white", fill=color)
            
    return region

def make_cnt_panel(tile_masks, color_map, width=200, pad=12, lh=18):
    """
    오른쪽 annotation 패널(클래스 목록 + 개수 + 메타정보)
    """
    font = load_font(14)
    # 클래스 카운트 (빈도 요약만 패널에 표시; 원한다면 제거해도 됨)
    cnt = Counter((m.get("cell_type") or "unknown").strip() for m in tile_masks)
    rows = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))

    # 패널 높이 계산(헤더/메타 + 클래스 행 수)
    height = pad*2 + lh* (4 + len(rows))
    panel = Image.new("RGB", (width, height), (255,255,255))
    d = ImageDraw.Draw(panel)

    y = pad
    y += 4
    for label, c in rows:
        color = hex_to_rgb(color_map.get(label, DEFAULT_COLOR))
        # 색상 박스
        d.rectangle([pad, y+4, pad+12, y+16], fill=color, outline=None)
        d.text((pad+18, y), f"{label} ({c})", font=font, fill=(0,0,0))
        y += lh

    return panel

def draw_annotation(dz, tile_masks, region_base, cfg, COLOR_MAP= COLOR_MAP, DEFAULT_COLOR = DEFAULT_COLOR):
    panel = make_cnt_panel(tile_masks, COLOR_MAP, width=200)

    # 좌측(region) + 우측(panel) 나란히 합치기
    region = draw_masks(dz, tile_masks, region_base, cfg, COLOR_MAP = COLOR_MAP, DEFAULT_COLOR = DEFAULT_COLOR)
    H = max(region.height, panel.height)
    canvas = Image.new("RGB", (region.width + panel.width, region.height), (255,255,255))
    canvas.paste(region, (0, 0))
    canvas.paste(panel, (region.width, 0))

    return canvas

def draw_together(region_base, canvas, show_fig = False):
    compare = Image.new("RGB", (region_base.width + canvas.width + 10, region_base.height), (255,255,255))
    compare.paste(region_base, (0, 0))
    compare.paste(canvas, (region_base.width+ 10, 0))
    if show_fig:
        plt.imshow(compare)
        plt.show()
    else:
        return compare
    
def save_png_current_res(img, path):
    # PNG 저장 시 dpi 메타가 int로 있으면 제거(또는 튜플화)
    if "dpi" in img.info:
        dpi_val = img.info["dpi"]
        if isinstance(dpi_val, int):
            # 1) 그냥 없애고 저장 (메타 제거)
            img.info.pop("dpi", None)
            # 2) 또는 튜플로 정상화해서 저장 원하면 아래 사용:
            # img.save(path, dpi=(dpi_val, dpi_val)); return
    img.save(path)
    
###################################
#Annotation
###################################

def tiles_covering_base_rect(slide, dz, level, x0, y0, w, h,
                             tile_size=224, overlap=0,
                             return_spans=False):
    """
    base(level-0) 사각형(x0,y0,w,h)을 덮는 DZ(level) 타일 범위 빠르게 계산.
    - return_spans=False: [(level, tx, ty), ...] 리스트
    - return_spans=True : [(ty, tx0, tx1), ...] 행 단위 span (더 빠른 순회에 유리)
    """
    # 1) DZ level 해상도 대비 base 배율 + bounds 오프셋
    W0, H0 = slide.dimensions
    Wd, Hd = dz.level_dimensions[level]
    sx, sy = W0 / float(Wd), H0 / float(Hd)  # dz픽셀→base픽셀 배율
    bx, by, _ , _  = get_bounds_offset(slide)

    # 2) base rect → DZ 픽셀 좌표 (오프셋 제거, 배율 반영)
    #    우측/하단 경계는 포함되도록 -eps(=1e-6) 대신 -1 픽셀 후 나눔
    dx0 = (x0 - bx) / sx
    dy0 = (y0 - by) / sy
    dx1 = (x0 + w - 1 - bx) / sx
    dy1 = (y0 + h - 1 - by) / sy

    # 3) DZ 타일 인덱스 범위 (step = tile_size - overlap)
    step = tile_size - overlap
    tx0 = int(math.floor(dx0 / step))
    ty0 = int(math.floor(dy0 / step))
    tx1 = int(math.floor(dx1 / step))
    ty1 = int(math.floor(dy1 / step))

    # 4) 그리드로 클램프
    tiles_w, tiles_h = dz.level_tiles[level]
    tx0 = max(0, min(tx0, tiles_w - 1))
    tx1 = max(0, min(tx1, tiles_w - 1))
    ty0 = max(0, min(ty0, tiles_h - 1))
    ty1 = max(0, min(ty1, tiles_h - 1))

    if return_spans:
        # 행 단위 span으로 반환: (ty, tx0, tx1)
        return [(ty, tx0, tx1) for ty in range(ty0, ty1 + 1)]
    else:
        # 개별 타일 인덱스 나열
        return [(tx, ty)
                for ty in range(ty0, ty1 + 1)
                for tx in range(tx0, tx1 + 1)]

def _ensure_group(groups_el: ET.Element, name: str, color_hex: str) -> None:
    for g in groups_el.findall("Group"):
        if g.attrib.get("Name") == name:
            return
    ET.SubElement(groups_el, "Group", Name=name, PartOfGroup="None", Color=color_hex)

def polygon_to_bbox_rect(points, pad=0.0):
    """
    points: [(x, y), ...]  # float/int 혼용 가능
    pad:   바운딩 박스 여유(픽셀). 0이면 정확히 외곽.
    return: [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]  시계방향
    """
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    xmin, xmax = min(xs) - pad, max(xs) + pad
    ymin, ymax = min(ys) - pad, max(ys) + pad
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

def replace_pol_to_rect(labels: Dict):
    label_rect = {}
    for cell_type, polys in labels.items():
        cell_dict = {}
        rects = []
        for poly in polys['polys']:
            rects.append(polygon_to_bbox_rect(poly, pad=0.0))
        cell_dict['color'] = labels[cell_type]['color']
        cell_dict['polys'] = rects
        label_rect[cell_type] = cell_dict
    return label_rect

def _add_polygon(annos_el: ET.Element, label: str, color_hex: str,
                 poly: List[Tuple[float, float]], anno_id: int) -> None:
    anno = ET.SubElement(annos_el, "Annotation",
                         Name=f"{label}_{anno_id}",
                         Type="Polygon",
                         PartOfGroup=label,
                         Color=color_hex)
    coords_el = ET.SubElement(anno, "Coordinates")
    for order, (x, y) in enumerate(poly):
        ET.SubElement(coords_el, 'Coordinate',
                      Order=str(order),
                      X=str(int(x)),
                      Y=str(int(y)))
        
def _local_to_abs_points(level_dims: Dict[int, Tuple[int,int]],
                         base_dims: Tuple[int,int],
                         lvl: int, tx: int, ty: int,
                         pts_local: List[Tuple[float,float]]) -> List[Tuple[float,float]]:
    """
    레벨 픽셀 좌표(타일 로컬) → 베이스 절대 좌표로 변환
    """
    W0, H0 = base_dims
    Wl, Hl = level_dims[lvl]
    sx, sy = W0 / float(Wl), H0 / float(Hl)
    TILE_SIZE = 224

    x_dz, y_dz = tx * TILE_SIZE, ty * TILE_SIZE
    x0 = int(round(x_dz * sx))
    y0 = int(round(y_dz * sy))

    return [(int(x0 + float(x) * sx), int(y0 + float(y) * sy)) for (x, y) in pts_local]

def _process_tile(args):
    """
    단일 타일 dict 처리(멀티프로세싱 대상)
    입력: (t, level_dims, base_dims, color_map, default_color)
    반환: [(label, color, poly_abs), ...] (해당 타일의 모든 mask 폴리곤)
    """
    t, level_dims, base_dims, color_map, default_color, contained = args
    tx  = int(t.get("x"))
    ty  = int(t.get("y"))
    lvl = int(t.get("level"))
    out = []
    masks = t.get("masks", [])
    if not masks:
        return out

    # 좌표 변환
    for m in masks:
        label = m.get("cell_type") or "Other"
        if label not in contained:
            continue
        color = color_map.get(label, default_color)
        pts_local = m.get("coordinates", [])
        if not pts_local or len(pts_local) < 3:
            continue

        pts_abs = _local_to_abs_points(level_dims, base_dims, lvl, tx, ty, pts_local)
        # ASAP은 최소 삼각형 이상이어야 의미 있음
        if len(pts_abs) >= 3:
            out.append((label, color, pts_abs))

    return out


def tiles_to_asap_xml(
    wsi_path: str,
    tiles: List[Dict],
    xml_out_path: str,
    color_map: Dict[str, str],
    roi_comment: str,
    contained: Optional[List[str]] = [],
    max_workers: int = max(1, (os.cpu_count() or 2) // 2),
) -> None:
    """
    타일 dict 리스트(각각에 level,x,y,width,height,masks 포함)를 ASAP XML로 변환
    - masks.coordinates: 타일 로컬 좌표(레벨 픽셀 기준)라고 가정
    - color_map: 라벨 → "#RRGGBB"
    """
    slide = openslide.OpenSlide(wsi_path)
    dz = DeepZoomGenerator(slide, tile_size=224, overlap=0, limit_bounds=False)

    # 변환에 필요한 메타
    level_dims = {lvl: dz.level_dimensions[lvl] for lvl in range(dz.level_count)}  # {lvl: (Wl,Hl)}
    base_dims  = slide.dimensions#  dz.level_dimensions[0]  # (W0,H0)

    # 병렬 처리
    label_rect: Dict[str, Dict[str, List[List[Tuple[float,float]]]]] = {}
    tasks = (
        (t, level_dims, base_dims, color_map, DEFAULT_COLOR, contained)
        for t in tiles
    )

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_tile, a) for a in tasks]
        for fut in as_completed(futures):
            for (label, color, poly) in fut.result():
                bucket = label_rect.setdefault(label, {"color": color, "polys": []})
                bucket["polys"].append(polygon_to_bbox_rect(poly))
        
    # ASAP XML 구성
    root = ET.Element("ASAP_Annotations")
    annos_el = ET.SubElement(root, "Annotations")
    groups_el = ET.SubElement(root, "AnnotationGroups")

    # 그룹 생성
    for label, meta in label_rect.items():
        _ensure_group(groups_el, label, meta["color"])

    # 폴리곤 추가(rect으로 추가)
    anno_id = 0
    for label, meta in label_rect.items():
        color = meta["color"]
        for poly in meta["polys"]:
            _add_polygon(annos_el, label, color, poly, anno_id)
            anno_id += 1

    # 저장

    # ---- ROI 정보 주석 추가 ----
    roi_comment = ET.Comment(roi_comment)
    root.insert(0, roi_comment)  # 루트 맨 앞에 삽입

    # xml_bytes = ET.tostring(root)#, encoding="utf-8")
    # xml_pretty = minidom.parseString(xml_bytes).toprettyxml(indent="  ")#, encoding="utf-8")
    xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    
    while os.path.exists(xml_out_path):
        xml_out_path = xml_out_path.replace('.xml','_1.xml')
        
    with open(xml_out_path, "w") as f:
        f.write(xml_string)
    
    print("ASAP XML saved:", xml_out_path)
    slide.close()
