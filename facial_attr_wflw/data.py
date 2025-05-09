# Auto‑generated from notebook functions/classes
def load_annotations(file_path):
def preprocess_images(data, image_folder, output_folder):
def split_data(data, test_size=0.2):
def calculate_mean_shape(data):
def align_face(image, src_landmarks, dst_landmarks):
def align_dataset(data, image_folder, output_folder, mean_shape):
def get_all_images_from_generator(generator):
def load_annotations(file_path):
def prepare_data(data, image_folder, augmented_annotations=None, augmented_folder=None, target_size=(224, 224), batch_size=2):
def augment_minority_classes(data, image_folder, augmented_folder, augmentation_factor=5, target_size=(224, 224)):
def generator_to_dataset(generator):
def predict(images, label_name):
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
def compute_metrics_np(y_true, y_pred, threshold):
def plot_correlation_matrix(y_true, y_pred, title, thresholds, model_output_names):
def predict_and_collect(generator, model):
class ModelEvaluator:
    def __init__(self, model_path, train_gen, val_gen):
    def compute_metrics(self, y_true, y_pred):
    def create_metrics_df(self):
    def save_metrics(self):
    def print_metrics(self):
    def identify_performance(self):
    def plot_correlation_matrices(self):
    def display_images(self, indices, images, title):
    def print_top_bottom_images(self):
def multi_task_model():
def attention_block(inputs, filters):
def attention_model(input_shape=(224, 224, 3)):
def get_embedding(img_path):
def deepFace_model(input_shape):
