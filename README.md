# BIG: Back In the Game of Creative AI

This is based on [DreamBooth in huggingface/diffusers](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth).

## Finetuning Model (Step-by-step)

You can finetune your own  or the metamodel wth your own style or object. For that:

1. prepare high-quality (at least 512x512) images of your style or object in a folder; quality over quantity but at least 5 images are recommended
2. specify the path to the images in `finetune_script.py`
3. specify what kind `category` your images are (e.g. dog, painting); this gives the model context on the learnt object
4. specify whether you want a private model, trained from pretrained, or let the metamodel learn your object
   - for your own model, set `target_model` to `'private'`
   - for the public metamodel, set `target_model` to `'meta'`
   - for the private metamodel, set `target_model` to `'private_meta'`
5. run `finetune_script.py`
6. you will get back an identifier, which can be used to create images of that object/style

## Generate Images (Step-by-step)

You can either use your own models or the metamodel to generate images. For that:

1. choose the identifier of the object/style you want to create
2. specify prompt with the identifier of the object/style you want to create 
   - note: category not needed as it's automatically fetched
3. specify whether you want to use your own model or use the metamodel or simply pretrained stable difusion
   - the own model will be identified by the identifier you got back from the finetuning and it has been solely trained to fit your style/object, set `target_model` to `'private'`
   - the metamodel has been trained to fit many styles from many users, set `target_model` to `'meta'`
   - for metamodel that has been trained to fit many styles & objects from only use, set `target_model` to `'private_meta'`
   - for the pretrained model, set `target_model` to `'pretrained'`
4. run `generate_script.py`
5. the generated image will be saved in the `generated_images` folder with the prompt and a timestamp as filename

## Listing Used Identifiers & Their Categories

In order to retrieve the identifiers for other objects/styles and their categories, you can execute `list_identifiers_script.py`.