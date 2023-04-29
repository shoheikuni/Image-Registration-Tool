# This code is originated from https://imagingsolution.net/program/python/tkinter/python_tkinter_image_viewer/
# and has been made some modifications.

import tkinter as tk
import tkinter.filedialog
from PIL import Image, ImageTk, ImageEnhance
import numpy as np
import os
import datetime
from common import *

class ImageAnnotation:
    def __init__(self, filepath):
        self.__filepath = filepath
        self.__annotationfilepath = filepath + '.anno'
        self.__rotation_angle_deg = 0 # filepathの画像は向きが間違っており、正しい向きに直すために必要な回転（反時計回り、単位は度）
        self.__enabled = True
        self.__dots = []
        self.__img = None
        if os.path.exists(self.__annotationfilepath):
            self.__read_annotation()

    def __read_annotation(self):
        with open(self.__annotationfilepath, "rt") as annotationfile:
            self.__rotation_angle_deg, self.__enabled, self.__dots = read_annotation(annotationfile)

    def __write_annotation(self):
        with open(self.__annotationfilepath, "wt") as annotationfile:
            write_annotation(annotationfile, self.__rotation_angle_deg, self.__enabled, self.__dots)

    @property
    def filepath(self):
        return self.__filepath

    @property
    def img(self):
        if not self.__img:
            self.__img = Image.open(self.__filepath)
        return self.__img

    @property
    def rotation_angle_deg(self):
        """ 画像の反時計回り回転角度 """
        return self.__rotation_angle_deg

    @property
    def image_rotation_matrix(self):
        """ 元画像（向きが不正）の座標系から正しい向きの画像の座標系への変換行列 """
        return image_rotation_matrix(self.img.width, self.img.height, self.rotation_angle_deg)


    def rotate_counterclockwise(self):
        self.__rotation_angle_deg = (self.__rotation_angle_deg + 90) % 360
        self.__write_annotation()

    def rotate_clockwise(self):
        self.__rotation_angle_deg = (self.__rotation_angle_deg + 360 - 90) % 360
        self.__write_annotation()

    @property
    def dots(self):
        return self.__dots
    @dots.setter
    def dots(self, val):
        self.__dots = val
        self.__write_annotation()

    @property
    def enabled(self):
        return self.__enabled

    @enabled.setter
    def enabled(self, val):
        self.__enabled = val
        self.__write_annotation()

def rotated_img(image_annotation):
    return image_annotation.img.rotate(image_annotation.rotation_angle_deg, expand=True)

def rotated_dots(image_annotation):
    return [matmul_ashomo(image_annotation.image_rotation_matrix, dot).astype(int) for dot in image_annotation.dots]


class Application(tk.Frame):

    class CanvasManipulator:
        def __init__(self, canvas, update_func):
            self.__canvas = canvas
            self.__target = None
            self.__update = update_func
            self.__reset_transform()

        def set_target(self, target):
            self.__target = target
            self.__zoom_fit(target.width, target.height)

        def handle_translate(self, event):
            if self.__target is None:
                return
            self.__translate(event.x - self.__old_event.x, event.y - self.__old_event.y)
            self.__update()
            self.__old_event = event

        def handle_start_translate(self, event):
            self.__old_event = event

        def handle_zoom_fit(self, event):
            if self.__target is None:
                return
            self.__zoom_fit(self.__target.width, self.__target.height)
            self.__update()

        def handle_zoom(self, event):
            if self.__target is None:
                return

            if (event.delta < 0):
                # 上に回転の場合、縮小
                self.__scale_at(0.8, event.x, event.y)
            else:
                # 下に回転の場合、拡大
                self.__scale_at(1.25, event.x, event.y)
            
            self.__update()

        # -------------------------------------------------------------------------------
        # ターゲット表示用アフィン変換
        # -------------------------------------------------------------------------------

        def __reset_transform(self):
            '''アフィン変換を初期化（スケール１、移動なし）に戻す'''
            self.target_to_canvas_matrix = np.eye(3) # 3x3の単位行列

        def __translate(self, offset_x, offset_y):
            ''' 平行移動 '''
            mat = np.eye(3) # 3x3の単位行列
            mat[0, 2] = float(offset_x)
            mat[1, 2] = float(offset_y)

            self.target_to_canvas_matrix = np.dot(mat, self.target_to_canvas_matrix)

        def __scale(self, scale:float):
            ''' 拡大縮小 '''
            mat = np.eye(3) # 単位行列
            mat[0, 0] = scale
            mat[1, 1] = scale

            self.target_to_canvas_matrix = np.dot(mat, self.target_to_canvas_matrix)

        def __scale_at(self, scale:float, cx:float, cy:float):
            ''' 座標(cx, cy)を中心に拡大縮小 '''

            # 原点へ移動
            self.__translate(-cx, -cy)
            # 拡大縮小
            self.__scale(scale)
            # 元に戻す
            self.__translate(cx, cy)

        def __zoom_fit(self, target_width, target_height):
            '''ターゲットをウィジェット全体に表示させる'''

            # キャンバスのサイズ
            canvas_width = self.__canvas.winfo_width()
            canvas_height = self.__canvas.winfo_height()

            if (target_width * target_height <= 0) or (canvas_width * canvas_height <= 0):
                return

            # アフィン変換の初期化
            self.__reset_transform()

            scale = 1.0
            offsetx = 0.0
            offsety = 0.0

            if (canvas_width * target_height) > (target_width * canvas_height):
                # ウィジェットが横長（ターゲットを縦に合わせる）
                scale = canvas_height / target_height
                # あまり部分の半分を中央に寄せる
                offsetx = (canvas_width - target_width * scale) / 2
            else:
                # ウィジェットが縦長（ターゲットを横に合わせる）
                scale = canvas_width / target_width
                # あまり部分の半分を中央に寄せる
                offsety = (canvas_height - target_height * scale) / 2

            # 拡大縮小
            self.__scale(scale)
            # あまり部分を中央に寄せる
            self.__translate(offsetx, offsety)


    def __init__(self, master=None):
        super().__init__(master)
        self.pack() 
 
        self.image_annotations = []
        self.current_image_index = 0
        self.current_img_dirty = True
        self.my_title = "Image Viewer"  # タイトル
        self.back_color = "#008B8B"     # 背景色
        self.brightness_factor = 1.0
        self.contrast_factor = 1.0

        self.points = []

        # ウィンドウの設定
        self.master.title(self.my_title)    # タイトル
        self.master.geometry("500x400")     # サイズ
 
        self.create_menu()   # メニューの作成
        self.create_widget() # ウィジェットの作成

    def menu_open_clicked(self, event=None):
        dirpath = tk.filedialog.askdirectory(initialdir = os.getcwd())
        if not dirpath:
            return

        os.chdir(dirpath)

        self.image_annotations = [ImageAnnotation(filepath) for filepath in get_imagefilenames_in(dirpath)]
        self.current_image_index = 0

        # 画像ファイルを設定する
        self.current_img_dirty = True
        self.set_current_image_to_canvas()

    def menu_quit_clicked(self):
        # ウィンドウを閉じる
        self.master.destroy() 

    # create_menuメソッドを定義
    def create_menu(self):
        self.menu_bar = tk.Menu(self) # Menuクラスからmenu_barインスタンスを生成
 
        self.file_menu = tk.Menu(self.menu_bar, tearoff = tk.OFF)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Open", command = self.menu_open_clicked, accelerator="Ctrl+O")
        self.file_menu.add_separator() # セパレーターを追加
        self.file_menu.add_command(label="Exit", command = self.menu_quit_clicked)

        self.menu_bar.bind_all("<Control-o>", self.menu_open_clicked) # ファイルを開くのショートカット(Ctrol-Oボタン)

        self.master.config(menu=self.menu_bar) # メニューバーの配置
 
    def create_widget(self):
        '''ウィジェットの作成'''

        # ステータスバー相当(親に追加)
        self.statusbar = tk.Frame(self.master)
        self.mouse_position = tk.Label(self.statusbar, relief = tk.SUNKEN, text="mouse position") # マウスの座標
        self.image_position = tk.Label(self.statusbar, relief = tk.SUNKEN, text="image position") # 画像の座標
        self.label_space = tk.Label(self.statusbar, relief = tk.SUNKEN)                           # 隙間を埋めるだけ
        self.image_info = tk.Label(self.statusbar, relief = tk.SUNKEN, text="image info")         # 画像情報
        self.mouse_position.pack(side=tk.LEFT)
        self.image_position.pack(side=tk.LEFT)
        self.label_space.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.image_info.pack(side=tk.RIGHT)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Canvas
        self.canvas = tk.Canvas(self.master, background= self.back_color)
        self.canvas.pack(expand=True,  fill=tk.BOTH)  # この両方でDock.Fillと同じ

        # キャンバス操作
        self.manipulator = self.CanvasManipulator(self.canvas, self.redraw_canvas)

        # マウスイベント
        self.master.bind("<Motion>", self.show_position)
        self.master.bind("<B2-Motion>", self.manipulator.handle_translate)
        self.master.bind("<Button-2>", self.manipulator.handle_start_translate)
        self.master.bind("<Double-Button-2>", self.manipulator.handle_zoom_fit)
        self.master.bind("<MouseWheel>", self.manipulator.handle_zoom)
        self.master.bind("<Button-1>", self.plot)
        self.master.bind("<Button-3>", self.deplot)
        self.master.bind("<Left>", self.switch_to_previous_image)
        self.master.bind("<Key-a>", self.switch_to_previous_image)
        self.master.bind("<Right>", self.switch_to_next_image)
        self.master.bind("<Key-d>", self.switch_to_next_image)
        self.master.bind("<Up>", self.rotate_image_counterclockwise)
        self.master.bind("<Key-w>", self.rotate_image_counterclockwise)
        self.master.bind("<Down>", self.rotate_image_clockwise)
        self.master.bind("<Key-s>", self.rotate_image_clockwise)
        self.master.bind("<space>", self.toggle_enabled)
        self.master.bind("<Shift-MouseWheel>", self.change_brightness)
        self.master.bind("<Control-MouseWheel>", self.change_contrast)

    def set_current_image_to_canvas(self):
        img = self.current_img

        if img is None:
            return

        # ウィンドウタイトルのファイル名を設定
        filename = os.path.basename(self.current_image_annotation.filepath)
        dt = exif_dateTimeOriginal(self.current_image_annotation.img)
        datetime_str = dt.strftime('%Y/%m/%d %H:%M:%S') if dt else "????-??-?? ??:??:??"
        self.master.title(self.my_title + " - " + filename + " - " + datetime_str)
        # ステータスバーに画像情報を表示する
        self.image_info["text"] = f"{img.format} : {img.width} x {img.height} {img.mode}"

        # 画像全体に表示するようにアフィン変換行列を設定
        self.manipulator.set_target(img)
        
        self.brightness_factor = 1.0
        self.contrast_factor = 1.0

        self.redraw_canvas()
        
    @property
    def current_image_annotation(self):
        return self.image_annotations[self.current_image_index] if self.image_annotations else None

    @property
    def current_img(self):
        if self.current_img_dirty:
            if self.current_image_annotation is None:
                self.__current_img = None
            else:
                self.__current_img = rotated_img(self.current_image_annotation)
                self.current_img_dirty = False
        return self.__current_img

    @property
    def current_dots(self):
        if self.current_image_annotation is None:
            return []
        return rotated_dots(self.current_image_annotation)

    def remove_dot_from_current_dots(self, index):
        if self.current_image_annotation is None:
            return
        dots = self.current_image_annotation.dots
        dots.pop(index)
        self.current_image_annotation.dots = dots

    def add_dot_to_current_dots(self, rotated_dot):
        if self.current_image_annotation is None:
            return
        inv_image_rotation_matrix = np.linalg.inv(self.current_image_annotation.image_rotation_matrix)
        dot = matmul_ashomo(inv_image_rotation_matrix, rotated_dot).astype(int)
        dots = self.current_image_annotation.dots
        dots.append(dot)
        self.current_image_annotation.dots = dots

    # -------------------------------------------------------------------------------
    # イベントハンドラ
    # -------------------------------------------------------------------------------

    def show_position(self, event):
        # マウス座標
        self.mouse_position["text"] = f"mouse(x, y) = ({event.x: 4d}, {event.y: 4d})"

        img_to_canvas_matrix = self.manipulator.target_to_canvas_matrix
        canvas_to_img_matrix = np.linalg.inv(img_to_canvas_matrix)

        canvas_event_pos = (event.x, event.y)
        x, y = matmul_ashomo(canvas_to_img_matrix, canvas_event_pos).astype(int)

        img = self.current_img
        if img is not None and x >= 0 and x < img.width and y >= 0 and y < img.height:
            # 輝度値の取得
            pixel = img.getpixel((x, y))
        else:
            pixel = '(-, -, -)'

        self.image_position["text"] = f"image({x: 4d}, {y: 4d}) = {pixel}"

    def plot(self, event):
        img_to_canvas_matrix = self.manipulator.target_to_canvas_matrix
        canvas_to_img_matrix = np.linalg.inv(img_to_canvas_matrix)

        canvas_event_pos = (event.x, event.y)
        img_event_pos = matmul_ashomo(canvas_to_img_matrix, canvas_event_pos).astype(int)

        # dotが（キャンバス上で）既存の点に近ければ、既存点を書き換える。そうでなければ、新たに点を生成する。
        threshold = 10
        dist = lambda p, q: np.linalg.norm(np.array(p) - np.array(q))
        for i, dot in enumerate(self.current_dots):
            canvas_dot = matmul_ashomo(img_to_canvas_matrix, dot)
            if dist(canvas_dot, canvas_event_pos) < threshold:
                self.remove_dot_from_current_dots(i)
        self.add_dot_to_current_dots(img_event_pos)

        self.redraw_canvas()

    def deplot(self, event):
        if not self.current_dots:
            return
        
        img_to_canvas_matrix = self.manipulator.target_to_canvas_matrix

        canvas_event_pos = (event.x, event.y)

        # dotが（キャンバス上で）既存の点に近ければ、既存点を書き換える。そうでなければ、新たに点を生成する。
        threshold = 10
        dist = lambda p, q: np.linalg.norm(np.array(p) - np.array(q))
        for i, dot in enumerate(self.current_dots):
            canvas_dot = matmul_ashomo(img_to_canvas_matrix, dot)
            if dist(canvas_dot, canvas_event_pos) < threshold:
                self.remove_dot_from_current_dots(i)

        self.redraw_canvas()

    def switch_to_previous_image(self, event):
        self.current_image_index = (self.current_image_index + len(self.image_annotations) - 1) % len(self.image_annotations)
        self.current_img_dirty = True
        self.set_current_image_to_canvas()

    def switch_to_next_image(self, event):
        self.current_image_index = (self.current_image_index + 1) % len(self.image_annotations)
        self.current_img_dirty = True
        self.set_current_image_to_canvas()

    def rotate_image_counterclockwise(self, event):
        if self.current_image_annotation is None:
            return
        
        self.current_image_annotation.rotate_counterclockwise()
        self.current_img_dirty = True
        self.redraw_canvas()

    def rotate_image_clockwise(self, event):
        if self.current_image_annotation is None:
            return
        
        self.current_image_annotation.rotate_clockwise()
        self.current_img_dirty = True
        self.redraw_canvas()

    def toggle_enabled(self, event):
        if self.current_image_annotation is None:
            return
        
        enabled = self.current_image_annotation.enabled
        self.current_image_annotation.enabled = not enabled
        self.redraw_canvas()

    def change_brightness(self, event):
        self.brightness_factor *= 1 + 0.1 * np.sign(event.delta)
        self.redraw_canvas()

    def change_contrast(self, event):
        self.contrast_factor *= 1 + 0.1 * np.sign(event.delta)
        self.redraw_canvas()

    # -------------------------------------------------------------------------------
    # 描画
    # -------------------------------------------------------------------------------

    def redraw_canvas(self):
        if self.current_img is None:
            return
        
        self.canvas.delete("all")

        # キャンバスのサイズ
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_to_canvas_matrix = self.manipulator.target_to_canvas_matrix
        canvas_to_img_matrix = np.linalg.inv(img_to_canvas_matrix)

        # PILの画像データをアフィン変換する
        final_img = self.current_img.transform(
                    (canvas_width, canvas_height),  # 出力サイズ
                    Image.AFFINE,                   # アフィン変換
                    tuple(canvas_to_img_matrix.flatten()),       # アフィン変換行列（出力→入力への変換行列）を一次元のタプルへ変換
                    Image.NEAREST,                  # 補間方法、ニアレストネイバー 
                    fillcolor= self.back_color
                    )

        final_img = ImageEnhance.Brightness(final_img).enhance(self.brightness_factor)
        final_img = ImageEnhance.Contrast(final_img).enhance(self.contrast_factor)
        
        # 表示用画像を保持
        self.__final_img = ImageTk.PhotoImage(image=final_img)

        # 画像の描画
        self.canvas.create_image(
                0, 0,               # 画像表示位置(左上の座標)
                anchor='nw',        # アンカー、左上が原点
                image=self.__final_img     # 表示画像データ
                )

        # 点をキャンバスに描く
        dots = self.current_dots
        dot_color = "green yellow" if len(dots) == 4 else "red"
        for dot in dots:
            canvas_dot = matmul_ashomo(img_to_canvas_matrix, dot).astype(int)
            radius = 5
            left, top = canvas_dot - radius
            right, bottom = canvas_dot + radius
            self.canvas.create_oval(left, top, right, bottom, fill=dot_color)

        if not self.current_image_annotation.enabled:
            left, top = matmul_ashomo(img_to_canvas_matrix, (0, 0)).astype(int)
            right, bottom = matmul_ashomo(img_to_canvas_matrix, (self.current_img.width, self.current_img.height)).astype(int)
            self.canvas.create_rectangle(left, top, right, bottom, outline="red", width=3)
            self.canvas.create_line(left, top, right, bottom, fill="red", width=3)
            self.canvas.create_line(right, top, left, bottom, fill="red", width=3)
        

if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
