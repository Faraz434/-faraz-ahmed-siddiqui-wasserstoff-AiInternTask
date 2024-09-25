def extract_objects(boxes, image):
    objects = []
    for i, box in enumerate(boxes):
        try:
            if len(box) < 6:
                logging.warning(f"Skipping invalid box with data: {box}")
                continue

            x1, y1, x2, y2, conf, class_id = box[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])  # Convert to integers

            object_image = image.crop((x1, y1, x2, y2))

            # Save to a temporary file
            temp_filename = tempfile.mktemp(suffix=".png")
            object_image.save(temp_filename)

            objects.append({
                'object_id': i,
                'filename': temp_filename,
                'bounding_box': (x1, y1, x2, y2),
                'confidence': conf,
                'class_id': class_id
            })
        except Exception as e:
            logging.error(f"Error processing box {i}: {e}")

    logging.info(f"Extracted {len(objects)} objects.")
    return objects

# Function to identify objects using auto-captioning
def identify_objects(objects):
    descriptions = []
    for obj in objects:
        try:
            image = Image.open(obj['filename'])
            caption = generate_caption(image)
            descriptions.append({
                'object_id': obj['object_id'],
                'description': caption
            })
            logging.info(f"Caption generated for object {obj['object_id']}: {caption}")
        except Exception as e:
            logging.error(f"Error in captioning object {obj['object_id']}: {e}")
            descriptions.append({
                'object_id': obj['object_id'],
                'description': "Captioning failed"
            })

    return descriptions

# Function to summarize attributes of the objects
def summarize_attributes(objects, descriptions):
    summary = []
    for obj in objects:
        description = next((item['description'] for item in descriptions if item['object_id'] == obj['object_id']), None)

        summary.append({
            'object_id': obj['object_id'],
            'bounding_box': obj['bounding_box'],
            'confidence': obj['confidence'],
            'class_id': obj['class_id'],
            'description': description
        })

    return summary

# Function to generate output image and CSV summary
def generate_output(image_path, summary):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    colors = ["red", "green", "blue", "yellow", "purple", "orange", "cyan", "magenta", "lime", "pink"]

    font_size = 20
    try:
        font = ImageFont.truetype("arialbd.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()
        logging.warning("Custom font 'arialbd.ttf' not found. Using default font.")

    segmented_images = []
    for obj in summary:
        x1, y1, x2, y2 = obj['bounding_box']
        color = colors[obj['object_id'] % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1 - font_size - 5), obj['description'], fill=color, font=font)

        # Save each segmented object image with label
        object_image_path = tempfile.mktemp(suffix=f"_{obj['object_id']}.png")
        image.crop((x1, y1, x2, y2)).save(object_image_path)
        segmented_images.append({
            'object_id': obj['object_id'],
            'image_path': object_image_path,
            'description': obj['description']
        })

    output_image_path = tempfile.mktemp(suffix=".png")
    image.save(output_image_path)

    df = pd.DataFrame(summary)
    output_csv_path = tempfile.mktemp(suffix=".csv")
    df.to_csv(output_csv_path, index=False)
    logging.info(f"Output saved as '{output_image_path}' and '{output_csv_path}'.")

    return output_image_path, output_csv_path, segmented_images

# Function to execute the full pipeline with parallel processing
def run_pipeline(image_path):
    try:
        boxes, image = segment_image(image_path)

        objects = extract_objects(boxes, image)
        descriptions = identify_objects(objects)

        summary = summarize_attributes(objects, descriptions)
        output_image_path, output_csv_path, segmented_images = generate_output(image_path, summary)
        logging.info("Pipeline completed successfully.")
        return output_image_path, output_csv_path, segmented_images
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        return None, None, None