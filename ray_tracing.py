import tkinter as tk
import pygame as pg
import numpy as np
import math

# константы 
EPS = 1e-4
RAY_MIN_DISTANCE = 0.0001
RAY_MAX_DISTANCE = 1.0e30

# цвета
WHITE = [255,255,255]
RED = [255,0,0]
GREEN = [0,255,0]
BLUE = [0,0,255]
YELLOW = [255,255,0]
PURPLE = [255,0,255]
CYAN = [0,255,255]

# нормализация вектора
def vect_normalize(vector: np.ndarray):
    return vector / np.linalg.norm(vector)

# изменить цвет
def col_intensivity(coef: float, color: list[int]) -> list[int]:
    return [ coef*i for i in color ]

# класс цвета
class Color():
    def __init__(self, r = 0, g = 0, b = 0):
        self.data = np.array([r, g, b])

    def from_array(array):
        r, g, b = array[0], array[1], array[2]
        return Color(r, g, b)

    def gray(w):
        return Color(w, w, w)

    # r, g, b from 0 to 255
    def unnormalized(r, g, b):
        return Color(r / 255, g / 255, b / 255)

    def __add__(self, other):
        if not isinstance(other, Color):
            raise TypeError("Cannot add color with not color")

        return Color.from_array(self.data + other.data)

    def __mul__(self, other):
        if not isinstance(other, Color):
            raise TypeError("Cannot multiply color with not color")

        return Color.from_array(self.data * other.data)
        
    def red(r=1.0):
        return Color(r, 0.0, 0.0)

    def green(g=1.0):
        return Color(0.0, g, 0.0)

    def blue(b=1.0):
        return Color(0.0, 0.0, b)

    def yellow(y=1.0):
        return Color(y, y, 0.0)

    def purple(p=1.0):
        return Color(p, 0.0, p)

    def turquoise(t=1.0):
        return Color(0.0, t, t)

# класс материала
class Material:
    def __init__(self, ambient_color: Color, diffuse_color: Color, specular_color: Color, 
                       shininess: float = 100.0, reflection: float = 0.0):
        self.ambient_color = ambient_color
        self.diffuse_color = diffuse_color
        self.specular_color = specular_color
        self.shininess = shininess
        self.reflection = reflection

# класс источника света
class PointLight:
    def __init__(self, position: np.ndarray, material: Material):
        self.position = position
        self.ambient_color = material.ambient_color
        self.diffuse_color = material.diffuse_color
        self.specular_color = material.specular_color

# класс луча
class Ray:
    def __init__(self, origin: np.ndarray, direction: np.ndarray, max_t = RAY_MAX_DISTANCE):
        self.origin = origin
        self.direction = vect_normalize(direction)
        self.max_t = max_t

    def ray_point(self, t: float):
        return self.origin + self.direction * t

    def copy(ray):
        return Ray(np.copy(ray.origin), np.copy(ray.direction), ray.max_t)

# класс пересечения 
class Intersection:
    def __init__(self, ray: Ray, shape = None, t: float = None):
        self.ray = ray
        self.t = t
        if t is None:
            self.t = ray.max_t
        self.intersected_shape = shape

    def material(self):
        return self.intersected_shape.material

    def normal(self):
        return self.intersected_shape.normal_at_point(self.position())

    def intersected(self):
        return (self.intersected_shape != None) and (self.t < self.ray.max_t)

    def position(self):
        return self.ray.ray_point(self.t)

    def __lt__(self, other):
        if not isinstance(other, Intersection):
            raise TypeError("Can compare only with Intersection")

        return self.t < other.t

# суперкласс фигур
class Shape:
    def __init__(self, mat: Material):
        self.material = mat

    def intersection(self, ray: Ray) -> Intersection:
        raise NotImplementedError("Intersect is not implemented")

    def normal_at_point(self, point: np.ndarray):
        raise NotImplementedError("Normal at point is not implemented")

# класс грани комнаты
class Plane(Shape):
    def __init__(self, position: np.ndarray, normal: np.ndarray, material: Material):
        self.position = position
        self.normal = normal
        super().__init__(material)

    def intersection(self, ray: Ray):
        ray_dir_dot_normal = np.dot(ray.direction, self.normal)
        if abs(ray_dir_dot_normal) < EPS:
            return None

        t = np.dot(self.position - ray.origin, self.normal) / ray_dir_dot_normal
        if (t < RAY_MIN_DISTANCE):
            return None
     
        return Intersection(ray, self, t)

    def normal_at_point(self, point: np.ndarray):
        return vect_normalize(self.normal)

# класс куба
class Cube(Shape):
    def __init__(self, center: np.ndarray, size_x, size_y, size_z, box_material):
        self.center = center
        self.sizes = np.array([size_x, size_y, size_z])
        self.left_bottom_corner = np.array([-size_x / 2, -size_y / 2, -size_z / 2])
        self.right_up_corner = np.array([size_x / 2, size_y / 2, size_z / 2])
        super().__init__(box_material)
    
    def intersection(self, ray: Ray) -> Intersection:
        local_ray = self.translated_ray(ray)
        tmin = (self.left_bottom_corner[0] - local_ray.origin[0]) / local_ray.direction[0]
        tmax = (self.right_up_corner[0] - local_ray.origin[0]) / local_ray.direction[0]

        if tmin > tmax:
            tmin, tmax = tmax, tmin
        
        tymin = (self.left_bottom_corner[1] - local_ray.origin[1]) / local_ray.direction[1]
        tymax = (self.right_up_corner[1] - local_ray.origin[1]) / local_ray.direction[1]

        if tymin > tymax:
            tymin, tymax = tymax, tymin

        if ((tmin > tymax) or (tymin > tmax)): 
            return None

        if (tymin > tmin): 
            tmin = tymin
    
        if (tymax < tmax):
            tmax = tymax

        tzmin = (self.left_bottom_corner[2] - local_ray.origin[2]) / local_ray.direction[2]
        tzmax = (self.right_up_corner[2] - local_ray.origin[2]) / local_ray.direction[2]

        if (tzmin > tzmax):
            tzmin, tzmax = tzmax, tzmin 
    
        if ((tmin > tzmax) or (tzmin > tmax)):
            return None 
    
        if (tzmin > tmin):
            tmin = tzmin
    
        if (tzmax < tmax): 
            tmax = tzmax

        if tmin >= tmax:
            return None

        if tmin > RAY_MIN_DISTANCE:
            return Intersection(ray, self, tmin)
        if tmax > RAY_MIN_DISTANCE:
            return Intersection(ray, self, tmax)
        
        return None

    def translated_ray(self, ray: Ray):
        translated_ray = Ray.copy(ray)
        translated_ray.origin = translated_ray.origin - self.center
        return translated_ray

    def step(self, edge: np.ndarray, vec: np.ndarray):
        return (vec >= edge) * 1.0

    def normal_at_point(self, point: np.ndarray):
        translated_point = point - self.center
        normal_dec = np.abs(translated_point) / self.sizes
        max_coordinate = np.max(normal_dec)
        
        normal_dec = normal_dec * (np.abs(normal_dec) >= max_coordinate)
        
        normal = normal_dec * translated_point
        normal = vect_normalize(normal)
        return normal

    def check_cached(self, point: np.ndarray):
        return np.min(np.abs(self.cached_point - point) < EPS)

# класс сферы
class Sphere(Shape):
    def __init__(self, center: np.ndarray, radius: float, material: Material):
        self.center = center
        self.radius = radius
        super().__init__(material)

    def intersection(self, ray: Ray):
        local_ray = self.translated_ray(ray)
        a = np.linalg.norm(local_ray.direction) ** 2
        b = 2 * np.dot(local_ray.direction, local_ray.origin)
        c = np.linalg.norm(local_ray.origin) ** 2 - self.radius ** 2
        ts = self.quadric_equation_solution(a, b, c)
        t = self.best_solution(ts)
        if t is None:
            return None
        return Intersection(ray, self, t)

    def quadric_equation_solution(self, a, b, c):
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return []
        
        t1 = (-b - math.sqrt(discriminant)) / (2 * a)
        t2 = (-b + math.sqrt(discriminant)) / (2 * a)
        if (t1 == t2):
            return [t1]
        
        return [t1, t2]


    def best_solution(self, ts):
        if len(ts) == 0:
            return None
        
        if len(ts) == 1:
            if ts[0] > RAY_MIN_DISTANCE:
                return ts[0]
            return None

        t1, t2 = sorted(ts)
        if t1 > RAY_MIN_DISTANCE:
            return t1
        if t2 > RAY_MIN_DISTANCE:
            return t2
        return None


    def translated_ray(self, ray: Ray):
        translated_ray = Ray.copy(ray)
        translated_ray.origin = translated_ray.origin - self.center
        return translated_ray

    def normal_at_point(self, point: np.ndarray):
        return vect_normalize(point - self.center)

# класс камеры
class Camera:
    def __init__(self, position: np.ndarray, target: np.ndarray, up_guide: np.ndarray, 
                 fov: float, aspect_ratio: float):
        self.position = position
        
        self.forward = vect_normalize(target - position)
        
        self.right = vect_normalize(np.cross(self.forward, up_guide))
        
        self.up = np.cross(self.right, self.forward)
        self.h = math.tan(fov)
        self.w = self.h * aspect_ratio

    def make_ray(self, x, y):
        direction = self.forward + (x * self.w * self.right + y * self.h * self.up)
        direction = direction / np.linalg.norm(direction)
        return Ray(self.position, direction)


# класс всей сцены
class Scene:
    shapes: list[Shape]
    
    def __init__(self, max_depth = 1):
        self.max_depth = max_depth
        self.shapes = []

    def add_shape(self, shape: Shape):
        self.shapes.append(shape)

    def add_light_source(self, light_source: PointLight):
        self.light_source = light_source

    def cast_ray(self, ray: Ray) -> Color:
        illumination = self.sum_illumination(ray)

        color = Color.from_array(np.clip(illumination, 0.0, 1.0))
        return color

    def sum_illumination(self, ray: Ray, depth=0):
        nearest_intersection = self.nearest_intersection(ray)
        if nearest_intersection is None:
            return np.zeros((3))
    
        illumination = self.calculate_illumination(ray, nearest_intersection)

        reflection = nearest_intersection.material().reflection

        if (depth == self.max_depth) or (reflection < EPS):
            return illumination

        reflected_ray = self.reflected(ray, nearest_intersection)
        return illumination + reflection * self.sum_illumination(reflected_ray, depth + 1)

    def nearest_intersection(self, ray: Ray):
        intersection = Intersection(ray)
        for shape in self.shapes:
            shape_intersection = shape.intersection(ray)
            if shape_intersection is None:
                continue
            if shape_intersection < intersection:
                intersection = shape_intersection
        
        if self.no_intersection(intersection):
            return None

        return intersection

    def no_intersection(self, intersection):
        return intersection == None or not intersection.intersected()

    def check_light(self, intersection):
        ray = self.ray_to_light(intersection)
        nearest_intersection = self.nearest_intersection(ray)
        return self.no_intersection(nearest_intersection) 

    def ray_to_light(self, intersection):
        pos = intersection.position()
        origin = pos + 1e-5 * intersection.normal()
        direction = self.light_source.position - origin
        distance_to_light = np.linalg.norm(direction)
        return Ray(origin, direction, distance_to_light)

    def calculate_illumination(self, ray: Ray, intersection: Intersection):
        direction_to_light = vect_normalize(self.light_source.position - intersection.position())
        direction_to_camera = vect_normalize(-ray.direction)
        
        normal = intersection.normal()
        material = intersection.material()
        
        # ambient
        ambient = (material.ambient_color * self.light_source.ambient_color).data
        
        if not self.check_light(intersection):
            return ambient
        
        # diffuse
        diffuse = (material.diffuse_color * self.light_source.diffuse_color).data 
        diffuse *= np.dot(direction_to_light, normal)
        # specular
        h = vect_normalize(direction_to_light + direction_to_camera)
        specular = (material.specular_color * self.light_source.specular_color).data
        specular *= np.dot(normal, h) ** (material.shininess / 4.0)

        illumination = ambient + diffuse + specular
        return illumination

    def reflected(self, ray: Ray, intersection: Intersection):
        normal = intersection.normal()
        reflected_ray_origin = intersection.position() + 1e-5 * normal
        reflected_ray_direction = ray.direction - 2 * np.dot(ray.direction, normal) * normal
        return Ray(reflected_ray_origin, reflected_ray_direction)

    def does_intesect(self, intersection: Intersection):
        for shape in self.shapes:
            if shape.intersect(intersection):
                return True
        return False


class App(tk.Tk):
    def __init__(self, width: int, height : int):
        super().__init__()
        self.title("Ray_tracing")
        self.geometry("200x100")
        tk.Label (self, text="Here is your ray tracing").pack()
        self.resizable(0,0)
        pg.display.set_caption("viewport")
        pg.init()

        self.width = width
        self.height = height

        #Building a scene
        self.scene = Scene()

        walls_shineness = 0.0

        walls_specular_color = Color.gray(0.0)

        floor_material = Material(Color.gray(0.1), Color.gray(0.5), walls_specular_color, walls_shineness)
        floor = Plane(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), floor_material)

        right_wall_material = Material(Color.blue(0.1), Color.blue(0.7), walls_specular_color, walls_shineness)
        right_wall = Plane(np.array([5.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), right_wall_material)

        left_wall_material = Material(Color.red(0.1), Color.red(0.7), walls_specular_color, walls_shineness)
        left_wall = Plane(np.array([-5.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), left_wall_material)

        back_wall_material = Material(Color.green(0.1), Color.green(0.5), walls_specular_color, walls_shineness, 0.8)
        back_wall = Plane(np.array([0.0, 0.0, 10.0]), np.array([0.0, 0.0, -1.0]), back_wall_material)
        roof = Plane(np.array([0.0, 10.0, 0.0]), np.array([0.0, -1.0, 0.0]), floor_material)

        front_wall = Plane(np.array([0.0, 0.0, -11.0]), np.array([0.0, 0.0, 1.0]), floor_material)

        self.scene.add_shape(floor)
        self.scene.add_shape(left_wall)
        self.scene.add_shape(right_wall)
        self.scene.add_shape(back_wall)
        self.scene.add_shape(front_wall)
        self.scene.add_shape(roof)

        box_material = Material(Color.turquoise(0.1), Color.turquoise(0.9), Color.gray(0.0), 0.0, 0.0)
        box = Cube(np.array([-2.0, 2.0, 6.0]), 3.0, 4.0, 3.0, box_material)
        sphere_material = Material(Color.yellow(0.1), Color.yellow(0.7), Color.gray(1.0))
        sphere = Sphere(np.array([1.0, 1.0, 3.0]), 1.0, sphere_material)
        reflecting_material = Material(Color.purple(0.1), Color.purple(0.7), Color.gray(0.0), 0.0, 0.6)
        reflecting_sphere = Sphere(np.array([2.5, 2.0, 6.0]), 2.0, reflecting_material)
    
        self.scene.add_shape(box)
        self.scene.add_shape(sphere)
        self.scene.add_shape(reflecting_sphere)

        light_material = Material(Color(1, 1, 1), Color(1, 1, 1), Color(1, 1, 1), 100.0)
        light = PointLight(np.array([0.0, 9.0, 6.0]), light_material)

        self.scene.add_light_source(light)

        self.camera = Camera(np.array([0.0, 5.0, -10.0]), np.array([0.0, 5.0, 2.0]), 
                    np.array([0.0, 1.0, 0.0]), 25 * math.pi / 180.0, self.width / self.height)

        #self.ray_trace(self.scene, self.camera)

    def run(self):
        self.canvas=pg.display.set_mode((self.width, self.height))

        self.ray_trace(self.scene, self.camera)

        pg.display.update()
        self.mainloop()

    # главная функция отрисовки
    def ray_trace(self,scene: Scene, camera: Camera):
        for x in range(self.width):
            for y in range(self.height):
                camera_x = 2.0 * x / self.width - 1.0
                camera_y = -2.0 * y / self.height + 1.0
                ray = camera.make_ray(camera_x, camera_y)
                color = scene.cast_ray(ray)
                if (x < 0 or x >= self.width):
                    raise ValueError("Invalid x")
                if (y < 0 or y >= self.height):
                    raise ValueError("Invalid y")

                self.canvas.set_at((x,y),pg.Color(255* color.data))

if __name__ == "__main__":
    app = App(640,360)
    app.run()