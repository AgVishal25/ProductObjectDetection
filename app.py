import streamlit as st
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import PIL.Image as Image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define transform outside the function for reuse
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((1024, 1024)),
    torchvision.transforms.RandomCrop((800, 800)),
    torchvision.transforms.ToTensor(),
])

def main():
    st.title("Image Uploader and Viewer")
    st.write("Upload an image and view it along with some information.")

    # File uploader widget
    file1 = st.file_uploader("Choose an image...", key="file1", type=["jpg", "jpeg", "png"])
    file2 = st.file_uploader("Choose an image...", key="file2", type=["jpg", "jpeg", "png"])
    predict_button = st.button("Predict")
    if predict_button:
        # Display the uploaded image
        image1 = Image.open(file1)
        image2 = Image.open(file2)
        # st.image(image1, caption='Uploaded Image', use_column_width=True)

        # Display some information about the image
        # img_info = get_image_info(image1)
        # st.write("**Image Information:**")
        # st.write(img_info)
        model = train_model(image1)
        prediction = detect_object(image2,model)
        check_product_presence(prediction)
        st.write("**Predictions:**")
        st.write(prediction)    

def train_model(image):
    # Initialize the bounding boxes for your product in the original image.
    bbox = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)  # Normalized [x1, y1, x2, y2] format

    # Prepare training dataset
    data_loader = [{'image': image, 'boxes': bbox, 'labels': torch.ones((1,), dtype=torch.int64)} for _ in range(100)]

    # Corrected Training Loop with device allocation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fasterrcnn_resnet50_fpn(pretrained=True, min_size=600, max_size=1333).to(device)
    model.train()

    # Simplify assuming you know how to perform an optimization step
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

    for batch in data_loader:
        images = [transform(batch['image']).to(device)]
        targets = [{'boxes': batch['boxes'].to(device), 'labels': batch['labels'].to(device)}]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()   # perform an optimization step (not shown here)
    return model

def detect_object(image, model):
    image = transform(image).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model([image])
    return prediction

def get_image_info(image):
    info = {}
    info['Format'] = image.format
    info['Mode'] = image.mode
    info['Size'] = f"{image.width}x{image.height}"
    return info

def check_product_presence(prediction, product_label=1, confidence_threshold=0.3):
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    for label, score in zip(labels, scores):
        if label.item() == product_label and score.item() >= confidence_threshold:
            return True
    return False

if __name__ == "__main__":
    main()
