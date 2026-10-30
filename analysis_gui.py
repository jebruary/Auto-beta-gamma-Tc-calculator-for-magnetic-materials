"""Qt adapter; all inference is delegated to critical_engine.run_pipeline."""
from critical_engine import PhysicsEngine, run_pipeline, save_json, json_safe


def run_gui_analysis(engine,config):
    """Synchronous adapter also used in interface parity tests."""
    return run_pipeline(engine,config)


def main():
    import sys
    import json
    from pathlib import Path
    try:
        from PyQt5 import QtCore, QtWidgets
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise SystemExit('Install requirements-optional.txt to use the GUI') from exc

    class Worker(QtCore.QThread):
        completed=QtCore.pyqtSignal(dict)
        def __init__(self,engine,cfg):
            super().__init__(); self.engine=engine; self.cfg=cfg
        def run(self):
            self.completed.emit(run_gui_analysis(self.engine,self.cfg))

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__(); self.setWindowTitle('Magnetic criticality analysis 0.2'); self.resize(1250,850)
            self.engine=PhysicsEngine(); self.paths=[]
            central=QtWidgets.QWidget(); self.setCentralWidget(central)
            layout=QtWidgets.QHBoxLayout(central); settings=QtWidgets.QWidget(); form=QtWidgets.QFormLayout(settings)
            settings.setMaximumWidth(380); layout.addWidget(settings)
            load=QtWidgets.QPushButton('Load isotherms'); load.clicked.connect(self.load); form.addRow(load)
            self.file_label=QtWidgets.QLabel('No files loaded'); self.file_label.setWordWrap(True); form.addRow(self.file_label)
            self.unit=QtWidgets.QComboBox(); self.unit.addItems(['Header units','T','mT','Oe','kOe','A/m','relative']); form.addRow('Applied field units',self.unit)
            self.munit=QtWidgets.QComboBox(); self.munit.addItems(['unspecified','A m2/kg','A/m']); form.addRow('Magnetisation units',self.munit)
            self.branch=QtWidgets.QComboBox(); self.branch.addItems(['auto','ascending','descending']); form.addRow('Measurement sweep',self.branch)
            self.reversible=QtWidgets.QCheckBox('Selected data are reversible and equilibrated'); form.addRow(self.reversible)
            self.auto=QtWidgets.QCheckBox('Select low field cutoff automatically'); form.addRow(self.auto)
            self.cut=QtWidgets.QDoubleSpinBox(); self.cut.setRange(0,1000); self.cut.setDecimals(5); form.addRow('Minimum field after conversion',self.cut)
            self.beta=QtWidgets.QDoubleSpinBox(); self.beta.setRange(.05,1.2); self.beta.setDecimals(4); self.beta.setValue(.365); form.addRow('Initial beta',self.beta)
            self.gamma=QtWidgets.QDoubleSpinBox(); self.gamma.setRange(.2,3.5); self.gamma.setDecimals(4); self.gamma.setValue(1.386); form.addRow('Initial gamma',self.gamma)
            self.tc=QtWidgets.QDoubleSpinBox(); self.tc.setRange(0,3000); self.tc.setDecimals(4); self.tc.setSpecialValueText('Estimate using KF'); form.addRow('Fixed Tc if known (K)',self.tc)
            self.boot=QtWidgets.QSpinBox(); self.boot.setRange(0,10000); self.boot.setSingleStep(100); form.addRow('Bootstrap replicates per mode',self.boot)
            self.mc=QtWidgets.QSpinBox(); self.mc.setRange(0,100000); self.mc.setSingleStep(1000); form.addRow('Sensitivity chain steps',self.mc)
            self.loss=QtWidgets.QDoubleSpinBox(); self.loss.setRange(.000001,100); self.loss.setDecimals(6); self.loss.setValue(1.); form.addRow('Loss temperature',self.loss)
            self.tc_mode=QtWidgets.QComboBox(); self.tc_mode.addItems(['fixed','joint']); form.addRow('Sensitivity Tc mode',self.tc_mode)
            self.gp=QtWidgets.QCheckBox('GP branch diagnostic'); self.eos=QtWidgets.QCheckBox('Arrott Noakes prediction diagnostic'); self.mce=QtWidgets.QCheckBox('Maxwell entropy on common field interval')
            for check in [self.gp,self.eos,self.mce]: form.addRow(check)
            self.run_button=QtWidgets.QPushButton('Run analysis'); self.run_button.clicked.connect(self.run_analysis); form.addRow(self.run_button)
            save=QtWidgets.QPushButton('Save results and plots'); save.clicked.connect(self.save); form.addRow(save)
            right=QtWidgets.QSplitter(QtCore.Qt.Vertical); layout.addWidget(right)
            self.figure=Figure(figsize=(9,5),layout='constrained'); self.canvas=FigureCanvasQTAgg(self.figure); right.addWidget(self.canvas)
            self.text=QtWidgets.QPlainTextEdit(); self.text.setReadOnly(True); right.addWidget(self.text)
        def load(self):
            self.paths,_=QtWidgets.QFileDialog.getOpenFileNames(self,'Isotherms','','Data (*.csv *.txt *.tsv *.dat)')
            self.file_label.setText(f'{len(self.paths)} files selected')
        def run_analysis(self):
            unit=None if self.unit.currentIndex()==0 else self.unit.currentText()
            self.engine.load_files(self.paths,field_unit=unit,magnetization_unit=self.munit.currentText())
            cfg=dict(field_unit=unit,magnetization_unit=self.munit.currentText(),branch=self.branch.currentText(),
                reversible=self.reversible.isChecked(),auto_cutoff=self.auto.isChecked(),H_min=self.cut.value(),
                init_beta=self.beta.value(),init_gamma=self.gamma.value(),tc_fixed=self.tc.value() or None,
                bootstrap=self.boot.value(),mcmc=self.mc.value(),loss_temperature=self.loss.value(),
                tc_mode=self.tc_mode.currentText(),use_gp=self.gp.isChecked(),use_eos=self.eos.isChecked(),use_mce=self.mce.isChecked())
            self.run_button.setEnabled(False); self.text.setPlainText('Analysis running')
            self.worker=Worker(self.engine,cfg); self.worker.completed.connect(self.finished); self.worker.start()
        def finished(self,result):
            self.run_button.setEnabled(True)
            keys=['status','failure','flags','beta','gamma','Tc','Tc_source','delta','H_min_T','universality_status','compatible_references']
            self.text.setPlainText(json.dumps(json_safe({k:result.get(k) for k in keys}),indent=2))
            if result['status']=='failed': return
            import numpy as np
            self.figure.clear(); axes=self.figure.subplots(1,2)
            for c in self.engine._scaled_curves(result['beta'],result['gamma'],result['Tc'],0):
                ax=axes[0 if c['side']<0 else 1]; ax.plot(c['x'],c['y'],'.-',ms=2,lw=.6)
            for ax,label in zip(axes,['Below Tc','Above Tc']):
                ax.set(title=label,xlabel='ln[(mu0 H / H0) / |epsilon|^(beta+gamma)]',ylabel='ln[(M/M0) / |epsilon|^beta]'); ax.grid(alpha=.2)
            self.canvas.draw()
        def save(self):
            directory=QtWidgets.QFileDialog.getExistingDirectory(self,'Save results')
            if not directory: return
            save_json(Path(directory)/'summary.json',self.engine.results)
            self.figure.savefig(Path(directory)/'scaling_diagnostics.pdf')
            self.text.appendPlainText('Results and plots saved')

    app=QtWidgets.QApplication(sys.argv); window=Window(); window.show()
    import os
    if os.environ.get('MAGNETIC_GUI_SMOKE')=='1':
        QtCore.QTimer.singleShot(250,app.quit)
    return app.exec_()


if __name__=='__main__':
    raise SystemExit(main())
