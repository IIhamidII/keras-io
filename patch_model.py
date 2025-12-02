import json

# Read the model.json file
with open('tfjs_letters_model/model.json', 'r') as f:
    model_json = json.load(f)

# Fix 1: Change batch_shape to batchInputShape
if 'modelTopology' in model_json:
    layers = model_json['modelTopology']['model_config']['config']['layers']
    if layers and 'config' in layers[0]:
        if 'batch_shape' in layers[0]['config']:
            layers[0]['config']['batchInputShape'] = layers[0]['config'].pop('batch_shape')

# Fix 2: Remove 'sequential/' prefix from weight names
if 'weightsManifest' in model_json:
    for manifest in model_json['weightsManifest']:
        for weight in manifest['weights']:
            if weight['name'].startswith('sequential/'):
                weight['name'] = weight['name'].replace('sequential/', '', 1)

# Write back
with open('tfjs_letters_model/model.json', 'w') as f:
    json.dump(model_json, f)

print("Model JSON patched successfully")
