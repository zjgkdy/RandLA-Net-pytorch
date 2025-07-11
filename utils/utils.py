import yaml

def load_yaml(path):
    try:
        print(f"\033[32m Opening arch config file {path}\033[0m")
        yaml_data = yaml.safe_load(open(path, 'r'))
        return yaml_data
    except Exception as e:
        print(e)
        print(f"Error opening {path} yaml file.")
        quit()

def get_learning_rate(optimizer):
    return optimizer.param_groups[0]['lr']


def compute_avg_grad_norm(model):
    total_norm = 0.0
    num_params = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            num_params += 1
    if num_params > 0:
        return (total_norm / num_params) ** 0.5
    else:
        return 0.0
