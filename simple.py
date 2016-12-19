
def avg(items):
    return sum(items) / float(len(items))

def flatten(iters):
    return [item for iter in iters for item in iter]
