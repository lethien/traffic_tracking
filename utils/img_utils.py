import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps
from six import BytesIO

def resize_image(image, new_width=256, new_height=256):
    pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
    pil_image = ImageOps.fit(image, (new_width, new_height), Image.ANTIALIAS)
    np.copyto(image, np.array(pil_image))    
    return image

def draw_roi_on_image(image, pts, thickness=4, color='#ff0000'):
    pil_image = Image.fromarray(np.uint8(image)).convert("RGB")
    draw = ImageDraw.Draw(pil_image)    
    draw.line(pts + [pts[0]],
            width=thickness,
            fill=color)
    np.copyto(image, np.array(pil_image))
    return image


def draw_bounding_box_on_image(image,
                               left,
                                top,
                                right,
                                bottom,
                               color,
                               font,
                               thickness=4,
                               display_str_list=()):
    """Adds a bounding box to an image."""    
    draw = ImageDraw.Draw(image)
    
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top),
                (left, top)],
                width=thickness,
                fill=color)

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = top + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in display_str_list[::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)],
                    fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                display_str,
                fill="black",
                font=font)
        text_bottom -= text_height - 2 * margin    

def draw_tracking_lines(image, paths, color, thickness=4):
    draw = ImageDraw.Draw(image)

    line_pts = []
    for left, top, right, bottom, _, _ in paths:
        line_pts.append(((left + right) / 2, (top + bottom) / 2))
    
    draw.line(line_pts,
            width=thickness,
            fill=color)

def draw_boxes_and_lines(image, trackers, track_dict, category_index):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())
    font = ImageFont.load_default()

    for left, top, right, bottom, track_id, obj_class in trackers:   
        display_str = "{}: {}".format(int(track_id), category_index[obj_class]['name'])
        color = colors[hash(str(track_id)) % len(colors)]
        image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
        draw_bounding_box_on_image(
            image_pil,
            left,
            top,
            right,
            bottom,
            color,
            font,
            display_str_list=[display_str])        
        draw_tracking_lines(
            image_pil,
            track_dict[track_id],
            color)
        np.copyto(image, np.array(image_pil))
    return image

def draw_boxes(image, boxes, class_names, scores, max_boxes=10, min_score=0.1):
    """Overlay labeled boxes on an image with formatted scores and label names."""
    colors = list(ImageColor.colormap.values())

    font = ImageFont.load_default()

    for i in range(min(boxes.shape[0], max_boxes)):
        if scores[i] >= min_score:
            image_pil = Image.fromarray(np.uint8(image)).convert("RGB")
            ymin, xmin, ymax, xmax = tuple(boxes[i])
            im_width, im_height = image_pil.size
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
            display_str = "{}: {}%".format(class_names[i],
                                            int(100 * scores[i]))
            color = colors[hash(class_names[i]) % len(colors)]
            draw_bounding_box_on_image(
                image_pil,
                left,
                top,
                right,
                bottom,
                color,
                font,
                display_str_list=[display_str])
            np.copyto(image, np.array(image_pil))
    return image