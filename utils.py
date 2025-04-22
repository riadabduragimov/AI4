def action_to_arrow(action):
    arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    return arrows.get(action, '.')
