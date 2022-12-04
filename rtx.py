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
color_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)
tonemapped_buffer = ti.Vector.field(3, dtype=ti.f32, shape=res)

@ti.kernel
def render():
    ...

@ti.kernel
def tonemap(accumulated: ti.f32):
    ...

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
