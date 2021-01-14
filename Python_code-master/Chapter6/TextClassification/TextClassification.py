from sklearn.datasets import fetch_20newsgroups
import ktrain
from ktrain import text

def preprocess_dataset():
    classes = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']
    train_data = fetch_20newsgroups(subset='train', categories=classes, shuffle=True, random_state=42)
    test_data = fetch_20newsgroups(subset='test', categories=classes, shuffle=True, random_state=42)
    
    return train_data.data,train_data.target, test_data.data, test_data.target, classes

def create_text_classification_model():
    MODEL_NAME = 'distilbert-base-uncased'
    train_features, train_labels, test_features, test_labels, train_classes = preprocess_dataset()
    trans = text.Transformer(MODEL_NAME, maxlen=500, classes=train_classes)
    train_preprocess = trans.preprocess_train(train_features, train_labels)
    val_preprocess = trans.preprocess_test(test_features, test_labels)
    model_data = trans.get_classifier()
    classification_model = ktrain.get_learner(model_data, train_data=train_preprocess, val_data=val_preprocess, batch_size=6)
    classification_model.fit_onecycle(5e-5, 4)
    return classification_model, trans

def predict_category(classification_model, trans, input_text):
    predictor = ktrain.get_predictor(classification_model.model, preproc=trans)
    results = predictor.predict(input_text)
    return results

if __name__=="__main__": 
    classification_model, trans = create_text_classification_model()
    input_text = 'Babies with down syndrome have an extra chromosome.'
    print(predict_category(classification_model, trans, input_text))

