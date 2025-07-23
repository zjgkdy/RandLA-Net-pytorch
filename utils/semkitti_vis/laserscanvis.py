#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import copy
import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt


class LaserScanVis:
    """Class that creates and handles a visualizer for a pointcloud"""

    def __init__(self, scan, scan_names, pred_label_names, gt_label_names, offset=0,
                 semantics=True, instances=False):
        self.pred_scan = scan
        self.gt_scan = copy.deepcopy(scan)
        self.scan_names = scan_names
        self.pred_label_names = pred_label_names
        self.gt_label_names = gt_label_names
        self.offset = offset
        self.total = len(self.scan_names)
        self.semantics = semantics
        self.instances = instances
        # sanity check
        if not self.semantics and self.instances:
            print("Instances are only allowed in when semantics=True")
            raise ValueError

        self.reset()
        self.update_scan()

    def reset(self):
        """ Reset. """
        # last key press (it should have a mutex, but visualization is not
        # safety critical, so let's do things wrong)
        self.action = "no"  # no, next, back, quit are the possibilities

        # new canvas prepared for visualizing data
        self.canvas = SceneCanvas(keys='interactive', show=True)
        # interface (n next, b back, q quit, very simple)
        self.canvas.events.key_press.connect(self.key_press)
        self.canvas.events.draw.connect(self.draw)
        # grid
        self.grid = self.canvas.central_widget.add_grid()

        # laserscan part
        self.gt_sem_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.gt_sem_view, 0, 0)
        self.gt_sem_vis = visuals.Markers()
        self.gt_sem_view.camera = 'turntable'
        self.gt_sem_view.add(self.gt_sem_vis)
        visuals.XYZAxis(parent=self.gt_sem_view.scene)
        
        # add semantics
        print("Using semantics in visualizer")
        self.pred_sem_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.canvas.scene)
        self.grid.add_widget(self.pred_sem_view, 0, 1)
        self.pred_sem_vis = visuals.Markers()
        self.pred_sem_view.camera = 'turntable'
        self.pred_sem_view.camera.link(self.gt_sem_view.camera)
        self.pred_sem_view.add(self.pred_sem_vis)
        visuals.XYZAxis(parent=self.pred_sem_view.scene)

        # img canvas size
        self.multiplier = 1
        self.canvas_W = 1024
        self.canvas_H = 64
        if self.semantics:
            self.multiplier += 1
        if self.instances:
            self.multiplier += 1

        # new canvas for img
        self.img_canvas = SceneCanvas(keys='interactive', show=True,
                                      size=(self.canvas_W, self.canvas_H * self.multiplier))
        # grid
        self.img_grid = self.img_canvas.central_widget.add_grid()
        # interface (n next, b back, q quit, very simple)
        self.img_canvas.events.key_press.connect(self.key_press)
        self.img_canvas.events.draw.connect(self.draw)

        # add semantics
        self.gt_sem_img_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.img_canvas.scene)
        self.img_grid.add_widget(self.gt_sem_img_view, 0, 0)
        self.gt_sem_img_vis = visuals.Image(cmap='viridis')
        self.gt_sem_img_view.add(self.gt_sem_img_vis)

        self.pred_sem_img_view = vispy.scene.widgets.ViewBox(border_color='white', parent=self.img_canvas.scene)
        self.img_grid.add_widget(self.pred_sem_img_view, 1, 0)
        self.pred_sem_img_vis = visuals.Image(cmap='viridis')
        self.pred_sem_img_view.add(self.pred_sem_img_vis)


    def get_mpl_colormap(self, cmap_name):
        cmap = plt.get_cmap(cmap_name)

        # Initialize the matplotlib color map
        sm = plt.cm.ScalarMappable(cmap=cmap)

        # Obtain linear color range
        color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

        return color_range.reshape(256, 3).astype(np.float32) / 255.0

    def update_scan(self):
        # first open data
        self.pred_scan.open_scan(self.scan_names[self.offset])
        self.pred_scan.open_label(self.pred_label_names[self.offset])
        self.pred_scan.colorize()
        self.gt_scan.open_scan(self.scan_names[self.offset])
        self.gt_scan.open_label(self.gt_label_names[self.offset])
        self.gt_scan.colorize()

        # then change names
        title = "scan " + str(self.offset)
        self.canvas.title = title
        self.img_canvas.title = title
        
        # plot semantics
        self.pred_sem_vis.set_data(self.pred_scan.points,
                              face_color=self.pred_scan.sem_label_color[..., ::-1],
                              edge_color=self.pred_scan.sem_label_color[..., ::-1],
                              size=1)

        self.gt_sem_vis.set_data(self.gt_scan.points,
                              face_color=self.gt_scan.sem_label_color[..., ::-1],
                              edge_color=self.gt_scan.sem_label_color[..., ::-1],
                              size=1)

        self.pred_sem_img_vis.set_data(self.pred_scan.proj_sem_color[..., ::-1])
        self.pred_sem_img_vis.update()

        self.gt_sem_img_vis.set_data(self.gt_scan.proj_sem_color[..., ::-1])
        self.gt_sem_img_vis.update()

    # interface
    def key_press(self, event):
        self.canvas.events.key_press.block()
        self.img_canvas.events.key_press.block()
        if event.key == 'N':
            self.offset += 1
            if self.offset >= self.total:
                self.offset = 0
            self.update_scan()
        elif event.key == 'B':
            self.offset -= 1
            if self.offset < 0:
                self.offset = self.total - 1
            self.update_scan()
        elif event.key == 'Q' or event.key == 'Escape':
            self.destroy()

    def draw(self, event):
        if self.canvas.events.key_press.blocked():
            self.canvas.events.key_press.unblock()
        if self.img_canvas.events.key_press.blocked():
            self.img_canvas.events.key_press.unblock()

    def destroy(self):
        # destroy the visualization
        self.canvas.close()
        self.img_canvas.close()
        vispy.app.quit()

    def run(self):
        vispy.app.run()
