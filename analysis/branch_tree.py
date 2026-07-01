import numpy as np

class Branch(object):
    def __init__(self, parent):
        self.parent = parent
        self.children = []
        self.all_coords = []
        self.start_vec = None
        self.end_vec = None
        self.angle_from_parent = 0
        self.length = None
        self.len_vec_calc = 5
        self.thickness = 0 
        self.end_start_vec = None
        self.start_start_vec = None
        self.parent_vector_around_trunck = None

    def add_child(self, obj):
        self.children.append(obj)

    def compute_start_vec(self):
        start_y, start_x = self.all_coords[0]
        if len(self.all_coords) < self.len_vec_calc:
            end_y,end_x = self.all_coords[-1]
        else:
            end_y,end_x = self.all_coords[self.len_vec_calc-1]
        self.end_start_vec = end_y, end_x
        self.start_start_vec = start_y,start_x
        self.start_vec = (end_y - start_y, end_x - start_x)

    def compute_angle_with_parent(self):
        if self.parent is None or self.parent_vector_around_trunck is None or self.start_vec is None:
            return
        dot = float(np.dot(self.parent_vector_around_trunck, self.start_vec))
        cross = float(np.cross(self.parent_vector_around_trunck, self.start_vec))
        self.angle_from_parent = float(np.degrees(np.arctan2(cross, dot)) % 360)

    def compute_length(self):
        total = 0.0
        for (y1,x1),(y2,x2) in zip(self.all_coords[:-1], self.all_coords[1:]):
            dy, dx = abs(y2-y1), abs(x2-x1)
            if dy == 0 and dx == 0:
                continue
            total += np.sqrt(2.0) if (dy and dx) else 1.0
        self.length = total
        return self.length

    def compute_thickness(self, all_coords_binary):
        self.thickness =  len(all_coords_binary)/self.length if self.length !=0 else 0
    
