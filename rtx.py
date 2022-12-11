import time
import tkinter as tk
import numpy as np
import taichi as ti

ti.init(arch=ti.gpu)

mat_none = 0
mat_lambertian = 1
mat_specular = 2
mat_glass = 3
mat_light = 4


class Mat(tk.Tk):
    mat_sphere = mat_lambertian
    mat_box = mat_lambertian
    mat_top = mat_lambertian
    mat_bottom = mat_lambertian
    mat_left = mat_lambertian
    mat_right = mat_lambertian
    mat_far = mat_lambertian

    def __init__(self):
        super().__init__()
        self.title("Select Materials")
        self.geometry("1000x200")
        self.resizable(False, False)
        self.create_widgets()

    def fill_lb(self, lb: tk.Listbox):
        lb.insert(0, "None")
        lb.insert(1, "Lambertian")
        lb.insert(2, "Specular")
        lb.insert(3, "Glass")
        lb.insert(4, "Light")

    def ok(self):
        Mat.mat_sphere = self.lb1.curselection()[0]
        Mat.mat_box = self.lb2.curselection()[0]
        Mat.mat_top = self.lb3.curselection()[0]
        Mat.mat_bottom = self.lb4.curselection()[0]
        Mat.mat_left = self.lb5.curselection()[0]
        Mat.mat_right = self.lb6.curselection()[0]
        Mat.mat_far = self.lb6.curselection()[0]
        self.destroy()

    def create_widgets(self):
        self.fr1 = tk.LabelFrame(self, text="Sphere")
        self.lb1 = tk.Listbox(self.fr1, exportselection=False)
        self.fill_lb(self.lb1)
        self.lb1.select_set(mat_lambertian)

        self.fr2 = tk.LabelFrame(self, text="Box")

        self.lb2 = tk.Listbox(self.fr2, exportselection=False)
        self.fill_lb(self.lb2)
        self.lb2.select_set(mat_lambertian)

        self.fr3 = tk.LabelFrame(self, text="Top")
        self.lb3 = tk.Listbox(self.fr3, exportselection=False)
        self.fill_lb(self.lb3)
        self.lb3.select_set(mat_lambertian)

        self.fr4 = tk.LabelFrame(self, text="Bottom")
        self.lb4 = tk.Listbox(self.fr4, exportselection=False)
        self.fill_lb(self.lb4)
        self.lb4.select_set(mat_lambertian)

        self.fr5 = tk.LabelFrame(self, text="Left")
        self.lb5 = tk.Listbox(self.fr5, exportselection=False)
        self.fill_lb(self.lb5)
        self.lb5.select_set(mat_lambertian)

        self.fr6 = tk.LabelFrame(self, text="Right")
        self.lb6 = tk.Listbox(self.fr6, exportselection=False)
        self.fill_lb(self.lb6)
        self.lb6.select_set(mat_lambertian)

        self.fr7 = tk.LabelFrame(self, text="Far")
        self.lb7 = tk.Listbox(self.fr7, exportselection=False)
        self.fill_lb(self.lb7)
        self.lb7.select_set(mat_lambertian)

        self.btn = tk.Button(self, text="OK", command=self.ok)

        self.fr1.pack(fill="both", expand="yes", side="left")
        self.lb1.pack()
        self.fr2.pack(fill="both", expand="yes", side="left")
        self.lb2.pack()
        self.fr3.pack(fill="both", expand="yes", side="left")
        self.lb3.pack()
        self.fr4.pack(fill="both", expand="yes", side="left")
        self.lb4.pack()
        self.fr5.pack(fill="both", expand="yes", side="left")
        self.lb5.pack()
        self.fr6.pack(fill="both", expand="yes", side="left")
        self.lb6.pack()
        self.fr7.pack(fill="both", expand="yes", side="left")
        self.lb7.pack()
        self.btn.pack(fill="both", expand="yes", side="bottom")

    def run(self):
        self.mainloop()

res = (800, 800)
camera_pos = ti.Vector([0.0,1.0,3.0])
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
tonemapped_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)

mat_none = 0
mat_lambertian = 1
mat_specular = 2
mat_glass = 3
mat_light = 4

inf = 1e10
eps = 1e-4

light_color = ti.Vector(list(np.array([0.9, 0.85, 0.7]))) # light color
light_normal = ti.Vector([0.0, -1.0, 0.0]) # light normal

light_y_pos = 2.0 - eps # light y position
light_x_min_pos = -0.25 # light x position
light_x_range = 0.5 # light x range
light_z_min_pos = 1.0 # light z position
light_z_range = 0.12 # light z range
light_area = light_x_range * light_z_range # light area
light_min_pos = ti.Vector([light_x_min_pos, light_y_pos, light_z_min_pos]) # light min position
light_max_pos = ti.Vector([ # light max position
    light_x_min_pos + light_x_range, light_y_pos,
    light_z_min_pos + light_z_range
])

@ti.func
def intersect_plane(pos, d, pt_on_plane, norm):
    dist = inf
    hit_pos = ti.Vector([0.0, 0.0, 0.0]) # hit position
    denom = d.dot(norm)
    if abs(denom) > eps: # if ray is not parallel to plane
        dist = norm.dot(pt_on_plane - pos) / denom # distance from ray origin to plane
        hit_pos = pos + d * dist # hit position
    return dist, hit_pos # return distance and hit position


@ti.func
def intersect_box(box_min, box_max,o,d):
    intersect = 1
    
    near_face = 0 # face of the near intersection
    near_is_max = 0 # whether the near intersection is a max face
    
    tmin = (box_min[0]-o[0])/d[0]
    tmax = (box_max[0]-o[0])/d[0]
    
    if tmin > tmax:
        tmin, tmax = tmax, tmin
        
    tymin = (box_min[1]-o[1])/d[1]
    tymax = (box_max[1]-o[1])/d[1]
    
    if tymin > tymax:
        tymin, tymax = tymax, tymin
        
    if tmin>tymax or tymin > tmax:
        intersect = 0
    
    if tymin > tmin:
        tmin = tymin
        near_face = 1
        
    if tymax < tmax:
        tmax = tymax
        
    tzmin = (box_min[2]-o[2])/d[2]
    tzmax = (box_max[2]-o[2])/d[2]
    
    if tzmin > tzmax:
        tzmin, tzmax = tzmax, tzmin
        
    if tmin > tzmax or tzmin > tmax:
        intersect = 0
        
    if tzmin > tmin:
        tmin = tzmin
        near_face = 2
        
    if tzmax < tmax:
        tmax = tzmax
    near_norm = ti.Vector([0.0, 0.0, 0.0])        
    if intersect: # if there is an intersection
        for i in ti.static(range(3)): # for each axis except the near face
            if near_face == i: # if the near intersection is on the i-th face
                near_norm[i] = -1 + near_is_max * 2 # set the normal of the near intersection
    return intersect, tmin, tmax, near_norm         

@ti.func
def intersect_light(pos, d, tmax): # tmax is the max distance to the light
    hit, t, far_t, near_norm = intersect_box(light_min_pos, light_max_pos,  pos, d) # intersect with the light box
    if hit and 0 < t < tmax: # if hit and the distance is smaller than tmax
        hit = 1 # hit the light
    else: # otherwise
        hit = 0 # miss the light
        t = inf # set the distance to inf
    return hit, t # return the hit status and the distance

@ti.func
def mat_mul_point(m, p):
    hp = ti.Vector([p[0], p[1], p[2], 1.0])
    hp = m @ hp
    hp /= hp[3]
    return ti.Vector([hp[0], hp[1], hp[2]])

@ti.func
def mat_mul_vec(m, v):
    hv = ti.Vector([v[0], v[1], v[2], 0.0])
    hv = m @ hv
    return ti.Vector([hv[0], hv[1], hv[2]])

def make_box_transform_matrices():
    rad = np.pi / 8.0
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    translate = np.array([
        [1, 0, 0, 0.3],
        [0, 1, 0, 0],
        [0, 0, 1, 0.7],
        [0, 0, 0, 1],
    ])
    m = translate @ rot
    m_inv = np.linalg.inv(m)
    m_inv_t = np.transpose(m_inv)
    return ti.Matrix(m_inv), ti.Matrix(m_inv_t)

def make_box_transform_matrices2():
    rad = np.pi / 8.0
    c, s = np.cos(rad), np.sin(rad)
    rot = np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])
    translate = np.array([
        [1, 0, 0, 0.15],
        [0, 1, 0, 0],
        [0, 0, 1, 0.8],
        [0, 0, 0, 1],
    ])
    m = translate @ rot
    m_inv = np.linalg.inv(m)
    m_inv_t = np.transpose(m_inv)
    return ti.Matrix(m_inv), ti.Matrix(m_inv_t)

# r box
BOX_MIN = ti.Vector([0.0, 0.0, 0.0]) # left bottom far corner
BOX_MAX = ti.Vector([0.5, 0.5, 0.5]) # right top near corner
BOX_M_INV, BOX_M_INV_T = make_box_transform_matrices()
# r2 box
BOX2_MIN = ti.Vector([0.0, 0.5, 0.0]) # left bottom far corner
BOX2_MAX = ti.Vector([0.5, 1.0, 0.5]) # right top near corner
BOX2_M_INV, BOX2_M_INV_T = make_box_transform_matrices2()

@ti.func
def intersect_box_transformed(box_min, box_max, o, d): # o, d are in world space
    # Transform the ray to the box's local space
    obj_o = mat_mul_point(BOX_M_INV, o) # o, d are in object space
    obj_d = mat_mul_vec(BOX_M_INV, d) # o, d are in object space
    intersect, near_t, _, near_norm = intersect_box(box_min, box_max, obj_o, obj_d)
    if intersect and 0 < near_t:
        # Transform the normal in the box's local space to world space
        near_norm = mat_mul_vec(BOX_M_INV_T, near_norm) # near_norm is in world space
    else: # no intersection
        intersect = 0
    return intersect, near_t, near_norm # near_t is in world space, near_norm is in world space
@ti.func
def intersect_box_transformed2(box_min, box_max, o, d): # o, d are in world space
    # Transform the ray to the box's local space
    obj_o = mat_mul_point(BOX2_M_INV, o) # o, d are in object space
    obj_d = mat_mul_vec(BOX2_M_INV, d) # o, d are in object space
    intersect, near_t, _, near_norm = intersect_box(box_min, box_max, obj_o, obj_d)
    if intersect and 0 < near_t:
        # Transform the normal in the box's local space to world space
        near_norm = mat_mul_vec(BOX2_M_INV_T, near_norm) # near_norm is in world space
    else: # no intersection
        intersect = 0
    return intersect, near_t, near_norm # near_t is in world space, near_norm is in world space

@ti.func
def intersect_scene(pos, ray_dir):
    closest,  normal = inf, ti.Vector.zero(ti.f32,3)
    c, mat = ti.Vector.zero(ti.f32,3), mat_none
    
    hit, cur_dist, pnorm = intersect_box_transformed(BOX_MIN, BOX_MAX, pos, ray_dir)  # intersect ray with box
    if hit and 0 < cur_dist < closest:  # light hit the box
        closest = cur_dist  # update the closest distance
        normal = pnorm  # update the normal
        c, mat = ti.Vector([0.8, 0.5, 0.4]), Mat.mat_box  # update the color and material
    hit, cur_dist, pnorm = intersect_box_transformed2(BOX2_MIN, BOX2_MAX, pos, ray_dir)  # intersect ray with box
    if hit and 0 < cur_dist < closest:  # light hit the box
        closest = cur_dist  # update the closest distance
        normal = pnorm  # update the normal
        c, mat = ti.Vector([0.2, 0.3, 0.7]), mat_lambertian  # update the color and material        
    # left
    pnorm = ti.Vector([1.0, 0.0, 0.0])  # normal of the plane
    cur_dist, _ = intersect_plane(pos, ray_dir, ti.Vector([-1.1, 0.0, 0.0]), pnorm)  # intersect ray with plane (left)
    if 0 < cur_dist < closest: # light hit the plane
        closest = cur_dist # update the closest distance
        normal = pnorm # update the normal
        c, mat = ti.Vector([0.15, 0.25, 0.75]), Mat.mat_left # update the color and material
    # right
    pnorm = ti.Vector([-1.0, 0.0, 0.0])
    cur_dist, _ = intersect_plane(pos, ray_dir, ti.Vector([1.1, 0.0, 0.0]), pnorm) # intersect ray with plane (right)
    if 0 < cur_dist < closest: # light hit the plane
        closest = cur_dist # update the closest distance
        normal = pnorm # update the normal
        c, mat = ti.Vector([0.65, 0.05, 0.05]), Mat.mat_right # update the color and material
    # bottom
    gray = ti.Vector([0.93, 0.93, 0.93]) # gray color
    pnorm = ti.Vector([0.0, 1.0, 0.0]) # normal of the plane
    cur_dist, _ = intersect_plane(pos, ray_dir, ti.Vector([0.0, 0.0, 0.0]), pnorm) # intersect ray with plane (bottom)
    if 0 < cur_dist < closest: # light hit the plane
        closest = cur_dist # update the closest distance
        normal = pnorm # update the normal
        c, mat = gray, Mat.mat_bottom # update the color and material    
    # top
    pnorm = ti.Vector([0.0, -1.0, 0.0]) # normal of the plane
    cur_dist, _ = intersect_plane(pos, ray_dir, ti.Vector([0.0, 2.0, 0.0]), pnorm) # intersect ray with plane (top)
    if 0 < cur_dist < closest: # light hit the plane
        closest = cur_dist # update the closest distance
        normal = pnorm # update the normal
        c, mat = gray, Mat.mat_top # update the color and material 
    # far
    pnorm = ti.Vector([0.0, 0.0, 1.0])
    cur_dist, _ = intersect_plane(pos, ray_dir, ti.Vector([0.0, 0.0, 0.0]), pnorm) # intersect ray with plane (far)
    if 0 < cur_dist < closest: # light hit the plane
        closest = cur_dist # update the closest distance
        normal = pnorm # update the normal
        c, mat = gray, Mat.mat_far # update the color and material               
    # light
    hit_l, cur_dist = intersect_light(pos, ray_dir, closest) # intersect ray with light
    if hit_l and 0 < cur_dist < closest: # light hit the light
        # technically speaking, no need to check the second term
        closest = cur_dist # update the closest distance
        normal = light_normal # update the normal
        c, mat = light_color, mat_light # update the color and material        
    
    return closest,normal,c,mat

@ti.func
def dot_or_zero(n, l):
    return ti.max(0.0, n.dot(l))

@ti.func
def compute_brdf_pdf(normal, sample_dir):  # cosine weighted
    return dot_or_zero(normal, sample_dir) / np.pi  # pdf

@ti.func
def sample_brdf(normal):
    r, theta = 0.0, 0.0  # r is radius, theta is angle
    sx = ti.random() * 2.0 - 1.0  # sx is sample x
    sy = ti.random() * 2.0 - 1.0  # sy is sample y
    if sx != 0 or sy != 0:
        if abs(sx) > abs(sy):
            r = sx
            theta = np.pi / 4 * (sy / sx)
        else:
            r = sy
            theta = np.pi / 4 * (2 - sx / sy)
    # Apply Malley's method to project disk to hemisphere
    u = ti.Vector([1.0, 0.0, 0.0])  # u is tangent vector
    if abs(normal[1]) < 1 - eps:  # if normal is not close to y axis
        u = normal.cross(ti.Vector([0.0, 1.0, 0.0]))  # u is tangent vector
    v = normal.cross(u)  # v is bitangent vector
    costt, sintt = ti.cos(theta), ti.sin(theta)  # costt is cos(theta), sintt is sin(theta)
    xy = (u * costt + v * sintt) * r  # xy is sample on disk
    zlen = ti.sqrt(ti.max(0.0, 1.0 - xy.dot(xy)))  # zlen is length of z axis
    return xy + zlen * normal  # return sample on hemisphere

@ti.func
def reflect(d, n):
    # Assuming |d| and |n| are normalized
    return d - 2.0 * d.dot(n) * n

@ti.func
def sample_ray_dir(indir, normal, hit_pos, mat):  # mat is material, not matrix
    u = ti.Vector([0.0, 0.0, 0.0]) # u is tangent vector
    pdf = 1.0 # pdf of the sampled direction
    if mat == mat_lambertian: # lambertian material
        u = sample_brdf(normal) # sample a direction from the brdf
        #pdf = ti.max(eps, compute_brdf_pdf(normal, u)) # compute the pdf of the sampled direction
    elif mat == mat_specular: # specular material
        u = reflect(indir, normal) # reflect the incident direction
    # elif mat == mat_glass: # glass material
    #     cos = indir.dot(normal) # cos is the cosine between the incident direction and the surface normal
    #     ni_over_nt = refr_idx # ni_over_nt is the ratio of the refractive indices
    #     outn = normal # outn is the surface normal
    #     if cos > 0.0: # if the incident direction is not behind the surface
    #         outn = -normal # outn is the surface normal
    #         cos = refr_idx * cos # cos is the cosine between the incident direction and the surface normal
    #     else: # if the incident direction is behind the surface
    #         ni_over_nt = 1.0 / refr_idx # ni_over_nt is the ratio of the refractive indices
    #         cos = -cos # cos is the cosine between the incident direction and the surface normal
    #     has_refr, refr_dir = refract(indir, outn, ni_over_nt) # refract the incident direction
    #     refl_prob = 1.0 # refl_prob is the probability of reflection (refraction if has_refr is true)
    #     if has_refr: # if the incident direction can be refracted
    #         refl_prob = schlick(cos, refr_idx) # refl_prob is the probability of reflection (refraction if has_refr is true)
    #     if ti.random() < refl_prob: # if the incident direction is reflected
    #         u = reflect(indir, normal) # reflect the incident direction
    #     else: # if the incident direction is refracted
    #         u = refr_dir # refract the incident direction
    return u.normalized(), pdf # return the sampled direction and the pdf of the sampled direction

@ti.kernel
def render():
    for u, v in color_buffer:
        ratio = res[0] / res[1]
        fov = 0.8
        pos = camera_pos
        ray_dir = ti.Vector([
            (2 * fov * u / res[1] -
             fov * ratio - 1e-5),
            (2 * fov * v / res[1] -
             fov - 1e-5),
            -1.0,
        ])
        
        ray_dir = ray_dir.normalized()
        
        color = ti.Vector([0.0,0.0,0.0])
        throughput = ti.Vector([1.0,1.0,1.0])
        
        depth = 0
        
        while depth < 2:
            closest, hit_normal, hit_color, mat = intersect_scene(pos,ray_dir)
            if mat == mat_none:
                break
            hit_pos = pos + closest*ray_dir
            hit_light = (mat == mat_light)
            
            if hit_light:
                color += throughput*light_color
                break
            
            depth += 1
            ray_dir, pdf = sample_ray_dir(ray_dir, hit_normal, hit_pos, mat)
            pos = hit_pos + eps*ray_dir           

            throughput *= hit_color               
        color_buffer[u,v] += color

@ti.kernel
def tonemap(accumulated: ti.f32):
    for i, j in tonemapped_buffer:
        tonemapped_buffer[i, j] = ti.sqrt(color_buffer[i, j] / accumulated *
                                          100.0)

def main():
    Mat().run()
    gui = ti.GUI('Cornell Box', res, fast_gui=True)
    gui.fps_limit = 300
    last_t = time.time()
    i = 0
    while gui.running:
        render()
        gui.get_event()
        if gui.is_pressed(ti.GUI.ESCAPE):
            break

        interval = 10
        if i % interval == 0:
            tonemap(i)
            print(
                f"{interval / (time.time() - last_t):.2f} samples/s ({i} iters)"
            )
            last_t = time.time()
            gui.set_image(tonemapped_buffer)
            gui.show()
        i += 1


if __name__ == '__main__':
    main()
