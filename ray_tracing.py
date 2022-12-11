import tkinter as tk
import pygame as pg
import numpy as np
from enum import Enum
import random

class Material(Enum):
    Common = 1
    Mirror = 2

class App(tk.Tk):
    W: int = 100
    H: int = 100
    
    camera = np.array([0.0,1.0,3.0])
    light = np.array([0.0,0.2,0.0])
    light_color = pg.Color([255,255,255,255])
    default_dist = 1000000
    
    left_wall_color = pg.Color([255,0,0,255])
    left_wall_material = Material.Mirror
    
    right_wall_color = pg.Color([0,255,0,255])
    right_wall_material = Material.Common    

    top_wall_color = pg.Color([200,200,200,255])
    top_wall_material = Material.Common  
    
    floor_wall_color = pg.Color([200,200,200,255])
    floor_wall_material = Material.Common  
    
    box_min = np.array([-0.7,2.0,-3.0])
    box_max = np.array([-0.4, 1.0, -1.0])
            
    def __init__(self):
        super().__init__()
        self.title("Ray-tracing")
        self.resizable(0, 0)
        self.geometry(f"{self.W+200}x{70}+0+0")
        pg.display.set_caption("Viewport")
        pg.init()
        random.seed(2)

        
    def run(self):
        self.canvas = pg.display.set_mode((self.W, self.H)) 
        self.draw()
        pg.display.update()
        self.mainloop()
    
    def intersect_wall(self,ray,cur_pos, wall_pos, wall_norm):
        dist = self.default_dist
        angle = np.dot(ray,wall_norm)
        
        if abs(angle) > 0:
            dist = np.dot(wall_norm,wall_pos - cur_pos)/angle
            
        return dist

    def intersect_box(self,box_min, box_max,o,d):
        intersect = 1
    
        near_face = 0 # face of the near intersection
    
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
        near_norm = np.array([0.0, 0.0, 0.0])        
        if intersect:
            for i in range(3):
                if near_face == i:
                    near_norm[i] = -1
        return intersect, tmin, tmax, near_norm  
    
    def check_intersection(self, ray, cur_pos, to_light=False):
        cur_dist = self.default_dist
        int_norm = np.array([0.0,0.0,0.0])
        res_color = pg.Color([0.0,0.0,0.0])
        res_mat = Material.Common
        
        dist = self.intersect_wall(ray,cur_pos,np.array([-2,0,0]),np.array([1,0,0]))
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([1.0,0.0,0.0])
            res_color = self.left_wall_color
            res_mat = self.left_wall_material
            
        dist = self.intersect_wall(ray,cur_pos,np.array([2,0,0]),np.array([-1,0,0]))
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([-1.0,0.0,0.0])
            res_color = self.right_wall_color
            res_mat = self.right_wall_material

        dist = self.intersect_wall(ray,cur_pos,np.array([0,2,0]),np.array([0,1,0]))
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([0.0,1.0,0.0])
            res_color = self.floor_wall_color
            res_mat = self.floor_wall_material

        dist = self.intersect_wall(ray,cur_pos,np.array([0,0,0]),np.array([0,-1,0]))
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([0.0,-1.0,0.0])
            res_color = self.top_wall_color
            res_mat = self.top_wall_material

        dist = self.intersect_wall(ray,cur_pos,np.array([0,0,-4]),np.array([0,0,1]))
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([0.0,0.0,1.0])
            res_color = self.top_wall_color
            res_mat = self.top_wall_material 
        
        intr, tmin, tmax, norm = self.intersect_box(self.box_min, self.box_max, cur_pos, ray)       
        if intr and 0<tmin<cur_dist:
            cur_dist = tmin
            int_norm = norm
            res_color = self.left_wall_color            
                               
        return cur_dist, int_norm, res_color, res_mat
    
    def gen_random_ray_from(self,normal):
        angle = random.uniform(0,1)/2#np.sin(np.radians(np.random.randn()%150+30))
        angle2 = angle*angle
        new_dir = normal
        if normal[0] == 1:
            new_dir[0]-=angle
            new_dir[1]+=angle
        if normal[1] == 1:
            new_dir[1]-=angle
            new_dir[0]+=angle          
        if normal[2] == 1:
            new_dir[2]-=angle
            new_dir[1]+=angle
        if normal[0] == -1:
            new_dir[0]+=angle
            new_dir[1]-=angle
        if normal[1] == -1:
            new_dir[1]+=angle
            new_dir[0]-=angle
        if normal[2] == -1:
            new_dir[2]+=angle
            new_dir[1]-=angle
        return new_dir 

    def reflect(self,d, n):
        return d - 2.0 * np.dot(d,n) * n
                                                                     
    def trace(self,ray,cur_pos,itert,color,acc):
        if itert > 6:
            return color
        
        int_dist, norm, int_color, int_mat = self.check_intersection(ray,cur_pos)
        if int_color[0] == 0 and int_color[1] == 0 and int_color[2] == 0:
            return color
        t = np.array(int_color)[0:3]
        tn = t/np.linalg.norm(t)         
        acc = acc * np.array([0.4,0.4,0.4])
        new_pos = cur_pos + int_dist * ray
        new_ray = ray
        
        if int_mat == Material.Common:       
            vecToLight = self.light - new_pos
            vecToLight = vecToLight / np.linalg.norm(vecToLight)
            visability = max(abs(np.dot(vecToLight,norm)),0.0)
            l_dist, _, _, _ = self.check_intersection(vecToLight, new_pos, True)
            if l_dist == self.default_dist:
                color += np.array(int_color if iter != 1 else self.light_color)[0:3]*visability*acc
            else: 
                color += t * (visability /(int_dist*int_dist))*acc
            new_ray = self.gen_random_ray_from(norm)
        elif int_mat == Material.Mirror:
            new_ray = self.reflect(ray,norm)
            
               
        return self.trace(new_ray,new_pos,itert+1,color,acc)
        
           
    def draw(self):
        for i in range(0,self.W):
            for j in range(0,self.H):
                
                ratio = self.W/self.H
                fov = 0.8
                
                start_dir = np.array([
                    2*fov*i/self.H - fov*ratio,
                    2*fov*j/self.H - fov,
                    -1.0
                ])
                
                start_dir = start_dir / np.linalg.norm(start_dir)
                res = self.trace(start_dir,self.camera,0,np.array([0.0,0.0,0.0]),np.array([1.0,1.0,1.0]))
                # if i+j % 10 == 0:
                #     res = np.sqrt(res/10 * 100)
                rcol = pg.Color(min(int(res[0]),255),min(255,int(res[1])),min(255,int(res[2])),255)
                self.canvas.set_at((i,j),rcol)
                # pg.display.update()
                print(i,j,sep=' ')        


        
if __name__ == "__main__":
    random.seed(2)
    app = App()
    app.run()