def create_model(model_name,args):
    if is_model(model_name):   
        package = model_name.split("/")[0]
        name = model_name.split("/")[1]
        imported = getattr(__import__("models"+package, fromlist=[name]), name)  
        loaded_model = imported.Model(args)
        return loaded_model
    else:
        raise Exception("Your model name is not in the list of defined_models")

    
def is_model(model_name):
    """ Check if a model name exists
    """
    return model_name in defined_models()

def defined_models():
    """ List of implemented models
    
    Returns:
        (list, string): Returns list of models
    """    
    models = ["feature_extractors/cnn_extractor",
            "transformers/vanilla_trasnformer",]
    return models