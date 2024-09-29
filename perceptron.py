#!/usr/bin/env python3

import tkinter as tk
import numpy as np
import pandas as pd

root = tk.Tk()
root.title("Perceptron Convergence Visualizer")
root.minsize(width=800, height=700)
root.configure(bg="gray")

points = []
points2 = []
current_class = 1
canvas_width = 600
canvas_height = 500
canvas_center_x = canvas_width // 2
canvas_center_y = canvas_height // 2
made_widget_1 = False
made_widget_2 = False

# all my widgets 
text_label = tk.Label(root, text="Press S to stop generating points for class 1.", font=("Helvetica", 16), bg="gray")
text_label2 = tk.Label(root, text="Press C to stop generating points for class 2.", font=("Helvetica", 16), bg="gray")


#### CALLBACK FUNCTIONS ####

def close_button_clicked():
    print("Program ended!")
    root.destroy()


def make_widget_1():
    global text_label
    text_label.pack(side="top")
    global made_widget_1
    made_widget_1 = True


def make_widget_2():
    text_label.pack_forget()
    global text_label2
    text_label2.pack(side="top")
    global made_widget_2
    made_widget_2 = True


def record_points(event):
    x, y = event.x - canvas_center_x, canvas_center_y - event.y 

    if current_class == 1:
        points.append((x, y))
        print(f"Point has been added at x: {x} and y: {y}")

        radius = 1.5
        canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill="red")
        if not made_widget_1:
            make_widget_1()

    elif current_class == 2:
        points2.append((x, y))
        print(f"Point2 has been added at x: {x} and y: {y}")
        radius = 1.5
        canvas.create_oval(event.x - radius, event.y - radius, event.x + radius, event.y + radius, fill="blue")
        if not made_widget_2:
            make_widget_2()


def stop_recording(event):
    root.unbind("<Button-1>")
    print("Point collection has been stopped!")
    global current_class
    current_class = 2
    print(points)
    start_making_points2.pack(side="bottom")


def stop_recording2(event):
    root.unbind("<Button-1>")
    print("Point collection for class 2 ended.")
    text_label2.pack_forget()
    point_gen_stopped_msg = tk.Label(root, text="All data points have been generated. Press the button below to begin the algorithm.", font=("Helvetica", 12), bg="gray")
    point_gen_stopped_msg.pack(side="top")
    root.after(5000, point_gen_stopped_msg.destroy)
    begin_perceptron_finder.pack(side="bottom")


def generate_points():
    print("Point generation started.")
    start_making_points.pack_forget()

    global canvas
    canvas = tk.Canvas(root, bg="white", width=600, height=500)
    canvas.pack(pady=50)
    canvas.create_line(0, canvas_center_y, canvas_width, canvas_center_y, fill="black", width=2, dash=(3, 3))
    canvas.create_line(canvas_center_x, 0, canvas_center_x, canvas_height, fill="black", width=2, dash=(3, 3))

    root.bind("<Button-1>", record_points)
    root.bind("<s>", stop_recording)
    root.bind("<c>", stop_recording2)


def generate_points2():
    print("Point generation for class 2 has started.")
    root.bind("<Button-1>", record_points)
    start_making_points2.pack_forget()


def plot_line(m, c):
    global canvas

    y_left = m * (-canvas_center_x) + c
    y_right = m * canvas_center_x + c

    y_left_canvas = canvas_center_y - y_left
    y_right_canvas = canvas_center_y - y_right

    line_id = canvas.create_line(0, y_left_canvas, canvas_width, y_right_canvas, fill="green", width=2)

    lower_side_id = canvas.create_polygon(
        0, canvas_height, 0, y_left_canvas, canvas_width, y_right_canvas, canvas_width, canvas_height,
        fill="lightblue", outline="")

    upper_side_id = canvas.create_polygon(
        0, 0, 0, y_left_canvas, canvas_width, y_right_canvas, canvas_width, 0, fill="lightcoral", outline="")

    canvas.tag_lower(lower_side_id)
    canvas.tag_lower(upper_side_id)

    return line_id, lower_side_id, upper_side_id


def begin_finding_perceptron():
    global canvas
    print("Clicked!")
    main()


#### PERCEPTRON IMPLEMENTATION ####
point_data = None
w = np.random.rand(3) * 0.1

def main():
    print("Main reached.")
    global begin_perceptron_finder
    begin_perceptron_finder.pack_forget()

    points_class1 = pd.DataFrame(points, columns=['x1', 'x2'])
    points_class1['y'] = 1  # red points

    points_class2 = pd.DataFrame(points2, columns=['x1', 'x2'])
    points_class2['y'] = -1  # blue points

    global point_data
    point_data = pd.concat([points_class1, points_class2], ignore_index=True)
    print("Data points are available in a pd df now.")
    optimise_perceptron()

count_widget = None # TODO 

def optimise_perceptron():

    global w
    hyperplane_id = None
    normal_vector_id = None
    lower_side_id = None
    upper_side_id = None

    while True:
        mistakes = 0
        w0 = w[0]
        w1 = w[1]
        w2 = w[2]
        m = -1 * w[0] / w[1]
        c = -w[2] / w[1]
        # draw_normal_vector(w0, w1, w2, m, c)

        if hyperplane_id:
            canvas.delete(hyperplane_id)
        if lower_side_id:
            canvas.delete(lower_side_id)
        if upper_side_id:
            canvas.delete(upper_side_id)
        if normal_vector_id:
            canvas.delete(normal_vector_id)

        hyperplane_id, lower_side_id, upper_side_id = plot_line(m, c)
        normal_vector_id = draw_normal_vector(w0, w1, w2, m, c)
        root.update()

        for row_index in range(len(point_data)):
            X = np.array([point_data.iloc[row_index]['x1'], point_data.iloc[row_index]['x2'], 1])
            prediction = np.sign(np.dot(w, X))
            y = point_data.iloc[row_index]['y']

            # print(f"point: {X}, correct label: {y}, model prediction: {prediction}")

            if prediction * y <= 0:
                print("Wrong prediction, updating weights.")
                w = w + y * X
                w0 = w[0]
                w1 = w[1]
                w2 = w[2]
                m = -1 * w[0] / w[1]
                c = -w[2] / w[1]
                print(f"updated weights: {w}")
                mistakes += 1

        if mistakes == 0:
            print("converged to an optimal value!\n")
            m = -1 * w[0] / w[1]
            c = -w[2] / w[1]
            plot_line(m, c)
            print(f"m: {m}, c: {c}")
            w0 = w[0]
            w1 = w[1]
            w2 = w[2]
            draw_normal_vector(w0, w1, w2, m, c)
            break


def draw_normal_vector(w0, w1, w2, m, c):

    scale_factor = 1

    x1_hyperplane = -canvas_center_x  
    y1_hyperplane = m * x1_hyperplane + c

    x2_hyperplane = canvas_center_x  
    y2_hyperplane = m * x2_hyperplane + c

    x_mid = (x1_hyperplane + x2_hyperplane) / 2 + canvas_center_x
    y_mid = (y1_hyperplane + y2_hyperplane) / 2 + canvas_center_y

    x_end = x_mid + w0 * scale_factor
    y_end = y_mid - w1 * scale_factor  
    normal_vec = canvas.create_line(x_mid, y_mid, x_end, y_end, fill="blue", arrow=tk.LAST, width=2)

    return normal_vec

   
##### EVENT GENERATORS/WIDGETS ####

close_button = tk.Button(root, text="X", command=close_button_clicked).pack(side="bottom", anchor="e", padx=0, pady=0)

start_making_points = tk.Button(root, text="Class 1", command=generate_points)
start_making_points.pack(side="bottom")

start_making_points2 = tk.Button(root, text="Class2", command=generate_points2)
begin_perceptron_finder = tk.Button(root, text="Find Optimal Boundary", command=begin_finding_perceptron)

root.mainloop()
