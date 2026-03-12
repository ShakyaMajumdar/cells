
from pathlib import Path
import javabridge
import bioformats

import cv2
import numpy as np
from scipy.signal import find_peaks, peak_widths
import pandas as pd

javabridge.start_vm(class_path=bioformats.JARS)
rootLoggerName = javabridge.get_static_field("org/slf4j/Logger","ROOT_LOGGER_NAME", "Ljava/lang/String;")
rootLogger = javabridge.static_call("org/slf4j/LoggerFactory","getLogger", "(Ljava/lang/String;)Lorg/slf4j/Logger;", rootLoggerName)
logLevel = javabridge.get_static_field("ch/qos/logback/classic/Level", "ERROR", "Lch/qos/logback/classic/Level;")
javabridge.call(rootLogger, "setLevel", "(Lch/qos/logback/classic/Level;)V", logLevel)

# RECT = (200, 200, 100, 100)
RECT = (300, 300, 400, 400)

PROJ_PROMINENCE_THRESH = 4
PROJ_DIST_THRESH = 5


def pol2cart(phi, rho, dst, center, maxRadius, flags):
    dstx, dsty = dst
    Kangle = dsty / (2 * np.pi)
    angleRad = phi / Kangle

    if flags & cv2.WARP_POLAR_LOG:
        Klog = dstx / np.log(maxRadius)
        magnitude = np.exp(rho / Klog)
    else:
        Klin = dstx / maxRadius
        magnitude = rho / Klin

    cx, cy = center
    x = (np.round(cx + magnitude * np.cos(angleRad)))
    y = (np.round(cy + magnitude * np.sin(angleRad)))
    return x, y

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask.astype(np.uint8)

for fold in "A3 A4 B1 B2 B3 B4".split():
    for i in range(1, 5):
        try:
            fp = f"20250908/20250429/{fold}/231_{fold}-{i}_001.vsi"
            print(fp)
            metadata =  bioformats.OMEXML(bioformats.get_omexml_metadata(path=fp))
            T = metadata.image().Pixels.SizeT
            W = metadata.image().Pixels.SizeX
            H = metadata.image().Pixels.SizeY
            physcale = metadata.image().Pixels.PhysicalSizeX

            timelapse = np.stack([bioformats.load_image(fp, t=t) for t in range(T)])

            current_proj_locations = set() 
            proj_counts = [0]*T
            retract_counts = [0]*T
            fname = Path(fp).stem
            for t in range(T):
                frame = timelapse[t, :, :]
                bgr_im = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                mask = np.zeros((W, H), np.uint8)
                cv2.grabCut(
                    img=bgr_im,
                    mask=mask,
                    rect=RECT,
                    bgdModel=np.zeros((1, 65), np.float64),
                    fgdModel=np.zeros((1, 65), np.float64),
                    iterCount=5,
                    mode=cv2.GC_INIT_WITH_RECT
                )
                mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
                closed = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel=create_circular_mask(9, 9))

                M = cv2.moments(closed)
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                polW, polH = (200, 360) 
                pol = cv2.warpPolar(closed, (polW, polH), (cx, cy), 200, cv2.WARP_POLAR_LOG)
                ds = polW - 1 - np.argmax(pol[:, ::-1], axis=1)
                new_proj_locations, _ = find_peaks(ds, prominence=PROJ_PROMINENCE_THRESH)
                
                # new projections
                for npl in new_proj_locations:
                    for cpl in current_proj_locations:
                        if abs(npl - cpl) <= PROJ_DIST_THRESH:
                            break
                    else:
                        proj_counts[t] += 1
                
                # retractions
                for cpl in current_proj_locations:
                    for npl in new_proj_locations:
                        if abs(npl - cpl) <= PROJ_DIST_THRESH:
                            break
                    else:
                        retract_counts[t] += 1

                
                current_proj_locations = new_proj_locations

                print(t, proj_counts[t], retract_counts[t])
                if t == T-1:
                    proj_w, proj_base_h, proj_base_l, proj_base_r = peak_widths(ds, new_proj_locations)
                    peak_x, peak_y = pol2cart(new_proj_locations, ds[new_proj_locations], (polW, polH), (cx, cy), 200, cv2.WARP_POLAR_LOG)
                    proj_base_l_x, proj_base_l_y = pol2cart(proj_base_l, proj_base_h, (polW, polH), (cx, cy), 200, cv2.WARP_POLAR_LOG)
                    proj_base_r_x, proj_base_r_y = pol2cart(proj_base_r, proj_base_h, (polW, polH), (cx, cy), 200, cv2.WARP_POLAR_LOG)
                    proj_base_w = np.hypot(proj_base_r_x - proj_base_l_x, proj_base_r_y - proj_base_l_y) * physcale
                    proj_base_cx, proj_base_cy = (proj_base_l_x+proj_base_r_x)/2, (proj_base_l_y + proj_base_r_y)/2
                    proj_h = np.hypot(peak_x - proj_base_cx, peak_y - proj_base_cy) * physcale
                    print(proj_base_w, proj_h)
                    with pd.ExcelWriter("output3.xlsx", mode="a", if_sheet_exists='replace') as writer:
                        pd.DataFrame(
                            zip(proj_base_w, proj_h), 
                            columns=["Projection Width (um)", "Projection Height (um)"]
                        ).to_excel(
                            writer,
                            sheet_name=fname,
                            index=False,
                        )
            
            with pd.ExcelWriter("output2.xlsx", mode="a", if_sheet_exists='replace') as writer:
                pd.DataFrame(
                    zip(range(T), proj_counts, retract_counts), 
                    columns=["No. of frame", "No. of projections", "No. of retractions"]
                ).to_excel(
                    writer,
                    sheet_name=fname,
                    index=False,
                )
        except Exception as err:
            print(err)
            continue

javabridge.kill_vm()
