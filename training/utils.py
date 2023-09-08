import inspect
import importlib
from functools import reduce

# Currently only supports lora adapters. More to come, hopefully.
adapter_types = ["lora"]

def get_compatible_module_type(adapter_type: adapter_types):
    """ Returns a list of all compatible adapter layer types, to facillitate automatic module construction. """
    if adapter_type == "lora":
        module = importlib.import_module('peft.tuners.lora.layer')
        class_names = [name for name, obj in inspect.getmembers(module, inspect.isclass) if obj.__module__ == module.__name__] # Get layer type names
        class_names += ["Linear8bitLt"] # Add bitsandbyes 8bit Linear layer. We should technically be pulling this from their library, too.
        return [name for name in class_names if all(sub not in name for sub in ["Lora", "lora"])] # Drop lora layer from the list

def get_weight_names(model):
    """ Gets all unique param names from a model. """
    unique_params = set()
    for name, param in model.named_parameters():
        name = name.rsplit(".", 2)
        name = name[-1] if name[-1] not in ["weight", "bias"] else name[-2]
        unique_params.add(name)
    return unique_params

def create_module_lists(model, adapter_type):
    """ Automatically selects all compatible adapter modules for transformation. """
    module_list = get_compatible_module_type(adapter_type)  # This contains the base classes as strings
    adapter_modules, manual_modules, unique_names = [], [], set()
    
    for name, param in model.named_parameters():
        module_name = name.rsplit(".", 1)
        if module_name[1] in ["weight", "bias"]: 
            name = module_name[0] # Drop the weight/bias terms from our path.
            module_name = name.rsplit(".", 1) # We need this to be consistent, so we'll reseed it.
        module_name = module_name[-1] # Now name is the full path, and module_name is the final parameter container.

        if module_name not in unique_names: # Only do this once per container name.

            unique_names.add(module_name)
            class_str = str(type(reduce(getattr, name.split('.'), model)))
            class_str = class_str.rstrip("'>")
            class_str = class_str.rsplit(".", 1)[-1]

            target_list = adapter_modules if class_str in module_list else manual_modules
            target_list.append(module_name) # This is just a pointer.

    return adapter_modules, manual_modules