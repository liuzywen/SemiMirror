from pathlib import Path

def mkdir(path, level=2, create_self=True):
    """ Make directory for this path,
    level is how many parent folders should be created.
    create_self is whether create path(if it is a file, it should not be created)

    e.g. : mkdir('/home/parent1/parent2/folder', level=3, create_self=False),
    it will first create parent1, then parent2, then folder.

    :param path: string
    :param level: int
    :param create_self: True or False
    :return:
    """
    p = Path(path)
    if create_self:
        paths = [p]
    else:
        paths = []
    level -= 1
    while level != 0:
        p = p.parent
        paths.append(p)
        level -= 1

    for p in paths[::-1]:
        p.mkdir(exist_ok=True)