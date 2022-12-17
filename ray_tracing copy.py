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
    
    box2_min = np.array([0.4,2.0,0.5])
    box2_max = np.array([0.7, 1.0, 1.0])
                
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

    def intersect_sphere(self,pos, d, center, radius):  # ray-sphere intersection
        T = pos - center # ray to sphere center
        A = 1.0 # A coefficient of quadratic equation
        B = 2.0 * T.dot(d) # B coefficient of quadratic equation
        C = T.dot(T) - radius * radius # C coefficient of quadratic equation
        delta = B * B - 4.0 * A * C # discriminant
        dist = self.default_dist # distance to intersection
        hit_pos = np.array([0.0, 0.0, 0.0]) # position of intersection
        if delta > -1e-4: # if delta is positive, there are two solutions
            delta = max(delta, 0) # if delta is negative, set it to zero
            sdelta = np.sqrt(delta) # square root of delta
            ratio = 0.5 / A # use 0.5 to avoid division in quadratic equation
            ret1 = ratio * (-B - sdelta) # first solution using quadratic formula
            dist = ret1 # set distance to first solution
            if dist < self.default_dist: # if first solution is valid
                # refinement
                old_dist = dist # save first solution
                new_pos = pos + d * dist # new position
                T = new_pos - center # new ray to sphere center
                A = 1.0 # new A coefficient of quadratic equation
                B = 2.0 * T.dot(d) # new B coefficient of quadratic equation
                C = T.dot(T) - radius * radius # new C coefficient of quadratic equation
                delta = B * B - 4 * A * C # new discriminant
                if delta > 0: # if new discriminant is positive
                    sdelta = np.sqrt(delta) # new square root of delta
                    ratio = 0.5 / A # new 1 / (2 * A) coefficient of quadratic equation
                    ret1 = ratio * (-B - sdelta) + old_dist # new first solution using quadratic formula
                    if ret1 > 0: # if new first solution is valid
                        dist = ret1 # set distance to new first solution
                        hit_pos = new_pos + ratio * (-B - sdelta) * d # set hit position
                else: # if new discriminant is negative
                    dist = self.default_dist # set distance to infinity
        return dist, hit_pos # return distance and hit position


    def intersect_box(self,box_min, box_max,o,d):
        intersect = 1
    
        near_t = -self.default_dist # t value of the near intersection
        far_t = self.default_dist # t value of the far intersection
        near_face = 0 # face of the near intersection

        for i in range(3): # for each axis
            if d[i] == 0: # ray parallel to the slab
                if o[i] < box_min[i] or o[i] > box_max[i]: # ray outside the slab
                    intersect = 0 # no intersection
            else: # ray not parallel to the slab
                i1 = (box_min[i] - o[i]) / d[i] # t value of the near intersection
                i2 = (box_max[i] - o[i]) / d[i] # t value of the far intersection

                new_far_t = max(i1, i2) # new far intersection is the farthest of the two
                new_near_t = min(i1, i2) # new near intersection is the nearest of the two
                new_near_is_max = i2 < i1 # new near intersection is a max face if i2 < i1

                far_t = min(new_far_t, far_t) # far intersection is the nearest of the two
                if new_near_t > near_t: # near intersection is the farthest of the two
                    near_t = new_near_t # update near intersection
                    near_face = int(i) # update near face

        near_norm = np.array([0.0, 0.0, 0.0]) # normal of the near intersection
        if near_t > far_t: # if the near intersection is behind the far intersection
            intersect = 0 # no intersection
        if intersect: # if there is an intersection
            for i in range(3): # for each axis except the near face
                if near_face == i: # if the near intersection is on the i-th face
                    near_norm[i] = -1 * (1 if i!=1 else -1)# set the normal of the near intersection
        return intersect, near_t, far_t, near_norm  

    eps = 1e-10
    
    def check_intersection(self, ray, cur_pos, to_light=False):
        cur_dist = self.default_dist
        int_norm = np.array([0.0,0.0,0.0])
        res_color = pg.Color([0.0,0.0,0.0])
        res_mat = Material.Common
        
        dist = self.intersect_wall(ray,cur_pos,np.array([-2,0,0]),np.array([1,0,0])) #l
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([1.0,0.0,0.0])
            res_color = self.left_wall_color
            res_mat = self.left_wall_material
            
        dist = self.intersect_wall(ray,cur_pos,np.array([2,0,0]),np.array([-1,0,0])) #r
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([-1.0,0.0,0.0])
            res_color = self.right_wall_color
            res_mat = Material.Common

        dist = self.intersect_wall(ray,cur_pos,np.array([0,2,0]),np.array([0,-1,0])) #f
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([0.0,-1.0,0.0])
            res_color = self.floor_wall_color
            res_mat = self.floor_wall_material

        dist = self.intersect_wall(ray,cur_pos,np.array([0,0,0]),np.array([0,1,0])) #t
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([0.0,1.0,0.0])
            res_color = self.top_wall_color
            res_mat = self.top_wall_material

        dist = self.intersect_wall(ray,cur_pos,np.array([0,0,-4]),np.array([0,0,1])) #front back
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([0.0,0.0,1.0])
            res_color = self.top_wall_color
            res_mat = Material.Mirror

        dist = self.intersect_wall(ray,cur_pos,np.array([0,0,4]),np.array([0,0,-1])) #cam back
        if 0<dist<cur_dist and not to_light:
            cur_dist = dist
            int_norm = np.array([0.0,0.0,-1.0])
            res_color = self.top_wall_color
            res_mat = Material.Common

        # intr, tmin, _, norm = self.intersect_box(self.box_min, self.box_max, cur_pos, ray)       
        # if intr and -self.eps<tmin<cur_dist:
        #     cur_dist = tmin
        #     int_norm = norm
        #     res_color = self.right_wall_color
        #     res_mat = Material.Common           
            
        # intr, tmin, _, norm = self.intersect_box(self.box2_min, self.box2_max, cur_pos, ray)       
        # if intr and -self.eps<tmin<cur_dist:
        #     cur_dist = tmin
        #     int_norm = norm
        #     res_color = self.left_wall_color
        #     res_mat = Material.Common        

        d, h = self.intersect_sphere(cur_pos,ray, np.array([0.4,1.0,-1]),0.2)

        if 0<d<cur_dist:
            cur_dist = d
            int_norm = h-np.array([0.4,1.0,1])
            int_norm = int_norm/np.linalg.norm(int_norm)
            res_color = self.left_wall_color
            res_mat = Material.Common                                    
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

    def reflect(self,ray, norm):
        return ray - 2.0 * np.dot(ray,norm) * norm
                                                                     
    def trace(self,ray,cur_pos,itert,color,acc):
        if itert > 5:
            return color
        
        int_dist, norm, int_color, int_mat = self.check_intersection(ray,cur_pos)
        if int_color[0] == 0 and int_color[1] == 0 and int_color[2] == 0:
            return color
        t = np.array(int_color)[0:3]
        tn = t/np.linalg.norm(t)         
        acc = acc 
        new_pos = cur_pos + int_dist * ray
        new_ray = ray
        
        vecToLight = self.light - new_pos
        vecToLight = vecToLight / np.linalg.norm(vecToLight)
        visability = max(abs(np.dot(vecToLight,norm)),0.0)
        l_dist, _, _, lmat = self.check_intersection(vecToLight, new_pos, True)
        
        if int_mat == Material.Common: 
            if lmat == Material.Mirror:
                color += np.array(self.light_color if iter==0 else int_color)[0:3]*visability*acc
            elif l_dist == self.default_dist:
                color += np.array(self.light_color if iter==0 else int_color)[0:3]*visability*acc
            # elif 0<l_dist: 
            #     color += t * (visability /(int_dist*int_dist))*acc
            # new_ray = self.gen_random_ray_from(norm)
            return color
        elif int_mat == Material.Mirror:
            if l_dist == self.default_dist or lmat == Material.Mirror:
                color += np.array(self.light_color if iter==0 else int_color)[0:3]*visability*acc * np.array([0.3,0.3,0.3])           
            new_ray = self.reflect(ray,norm)
            
               
        return self.trace(new_ray,new_pos,itert+1,color,acc* np.array([0.1,0.1,0.1]))
        
           
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
                if i == 100 and j == 100:
                    print('d')
                start_dir = start_dir / np.linalg.norm(start_dir)
                res = self.trace(start_dir,self.camera,0,np.array([0.0,0.0,0.0]),np.array([1.0,1.0,1.0]))
                # if i+j % 10 == 0:
                #     res = np.sqrt(res/10 * 100)
                rcol = pg.Color(min(int(res[0]),255),min(255,int(res[1])),min(255,int(res[2])),255)
                self.canvas.set_at((i,j),rcol)
                # pg.display.update()
                print(i,j,sep=' ')        


        
if __name__ == "__main__":
    app = App()
    app.run()