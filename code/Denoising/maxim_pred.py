import tensorflow as tf

def denoise(model, img):
    # Read the input image
#input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_img= img.astype(np.float32)/ 255.0
    # Process the image
    processed_img = _process_image(input_img, model)

    return processed_img

def _process_image(input_img, model):
    # Resize the input image
    image = tf.convert_to_tensor(input_img)
    
    # Handle multi-stage outputs, obtain the last scale output of the last stage
    preds = model.predict(tf.expand_dims(image, axis=0))
    if isinstance(preds,list):
        preds = preds[-1]
        if isinstance(preds,list):
            preds = preds[-1]
    preds = np.array(preds[0], np.float32)*255.0
    preds = (preds - np.min(preds))/(np.max(preds)- np.min(preds)) * 255
    preds = preds.astype(np.uint8)
    
    return preds

