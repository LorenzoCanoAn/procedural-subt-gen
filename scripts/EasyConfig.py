import base64
import sys
from enum import Enum

import yaml
from PyQt5.QtCore import Qt, QRectF, QRect
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget, QVBoxLayout, QHBoxLayout, \
    QFileDialog, QLineEdit, QLabel, QDialog, QCheckBox, QDialogButtonBox, QComboBox, QFrame, QSizePolicy, QTreeWidget, QTreeWidgetItem, QAbstractItemView, \
    QScrollArea, QHeaderView, QTextEdit


class EasyConfig:


    class Dialog(QDialog):

        def get_expanded(self):
            res = []

            def traver(node):
                res.append(1 if node.isExpanded() else 0)
                for i in range(node.childCount()):
                    traver(node.child(i))

            traver(self.list.invisibleRootItem())
            return res

        def set_expanded(self, val):
            def traver(node, vec):
                if len(vec) > 0:
                    node.setExpanded(vec.pop() == 1)
                    for i in range(node.childCount()):
                        traver(node.child(i), vec)

            val.reverse()
            traver(self.list.invisibleRootItem(), val)

        class String(QWidget):
            def __init__(self, name, value, pretty, extra=None):
                super().__init__(None)
                self.name = name
                self.pretty = pretty
                self.layout = QHBoxLayout()
                ql = QLabel(self.pretty)
                ql.setMinimumWidth(100)
                # self.layout.addWidget(ql)
                self.add_specific(value, extra)
                self.layout.setContentsMargins(5, 5, 5, 0)
                self.setLayout(self.layout)
                ql.setAlignment(Qt.AlignLeft)

            def add_specific(self, value, extra):
                self.ed = QLineEdit()
                if value is not None:
                    self.ed.setText(str(value))
                    self.ed.home(True)
                    self.ed.setSelection(0,0)
                self.layout.addWidget(self.ed)

            def get_value(self):
                return self.ed.text() if self.ed.text() != "" else None

            def set_value(self, value):
                return self.ed.setText(str(value))

            def get_name(self):
                return self.name

        class EditBox(String):
            def add_specific(self, value, extra):
                self.ed = QTextEdit()
                self.ed.setMaximumHeight(80)
                if value is not None:
                    self.ed.setText(str(value))
                self.layout.addWidget(self.ed)

            def set_value(self, value):
                return self.ed.setText(str(value).replace("&&", "\n"))

            def get_value(self):
                text = self.ed.toPlainText().replace("\n", "&&")
                return text if text != "" else None

        class Password(String):
            def __init__(self, name, value, pretty=None, extra=None):
                value = base64.decodebytes(value.encode()).decode() if value else None
                super().__init__(name, value, pretty, extra)
                self.ed.setEchoMode(QLineEdit.Password)
                # self.ed.setText("jjj")

            def get_value(self):
                if self.ed.text() != "":
                    return base64.encodebytes(self.ed.text().encode()).decode().replace('\n', '')
                else:
                    return None

        class ComboBox(String):
            def add_specific(self, value, extra):
                self.cb = QComboBox()
                self.cb.addItems(extra)
                self.layout.addWidget(self.cb, stretch=2)
                self.cb.setCurrentText(value)

            def get_value(self):
                return self.cb.currentText() if self.cb.currentText() != "" else None

            def set_value(self, value):
                return self.cb.setCurrentText(value)

        class Integer(String):
            def get_value(self):
                return int(self.ed.text()) if self.ed.text().isnumeric() else None

        class Float(String):
            def get_value(self):
                return float(self.ed.text()) if self.ed.text().replace(".", "").isnumeric() else None

        class Checkbox(QWidget):
            def __init__(self, name, value, pretty):
                super().__init__(None)
                self.name = name
                self.pretty = pretty
                self.layout = QHBoxLayout()
                self.cb = QCheckBox()
                self.layout.setAlignment(Qt.AlignLeft)
                # self.cb.setLayoutDirection(Qt.RightToLeft)
                ql = QLabel(pretty)
                # self.layout.addWidget(ql)
                ql.setMinimumWidth(100)

                self.layout.addWidget(self.cb)
                self.layout.setContentsMargins(5, 9, 5, 0)
                self.setLayout(self.layout)
                if value is not None:
                    self.cb.setChecked(value)

            def get_name(self):
                return self.name

            def get_value(self):
                return int(self.cb.isChecked())

            def set_value(self, value):
                self.cb.setChecked(value > 0)

        class File(String):
            def __init__(self, name, value, pretty, extra, is_save_dialog = False, is_folder_choice = False):
                super().__init__(name, value, pretty)
                self.btn = QPushButton()
                self.btn.setIcon(QIcon.fromTheme("document-open"))
                self.btn_discard = QPushButton()
                self.btn_discard.setMaximumWidth(25)
                self.btn_discard.setIcon(QIcon.fromTheme("window-close"))
                self.btn_discard.clicked.connect(self.discard)
                self.btn.setMaximumWidth(30)
                self.btn.clicked.connect(self.open_file)
                self.layout.addWidget(self.btn)
                self.layout.addWidget(self.btn_discard)
                self.ed.setReadOnly(True)
                self.is_save_dialog = is_save_dialog
                self.is_folder_choice = is_folder_choice
                self.extra = extra

            def discard(self):
                self.ed.setText("")

            def open_file(self):
                if self.is_save_dialog:
                    file_name, _ = QFileDialog.getSaveFileName(self, "Open " + self.extra.upper() + " Document", "",
                                                               self.extra.upper() + " Files (*." + self.extra + ")")
                elif self.is_folder_choice:
                    file_name = QFileDialog.getExistingDirectory(self, "Select Directory", "")
                else:
                    file_name, _ = QFileDialog.getOpenFileName(self, "Save " + self.extra.upper() + " Document", "",
                                                               self.extra.upper() + " Files (*." + self.extra + ")")
                if file_name != "":
                    self.ed.setText(file_name)

        def __init__(self, dic):
            super().__init__(None)
            self.setWindowTitle("EasyConfig")
            layout = QVBoxLayout(self)
            self.list = QTreeWidget()
            # self.list.setStyleSheet('background: palette(window)')
            self.list.header().setVisible(False)
            self.list.setSelectionMode(QAbstractItemView.NoSelection)
            self.list.setColumnCount(2)
            self.widgets = []

            self.setMinimumHeight(300)
            #self.list.setMinimumWidth(500)

            scroll = QScrollArea()
            scroll.setWidget(self.list)
            scroll.setWidgetResizable(True)

            layout.addWidget(scroll)
            dic.trigger(self.list, self.list.invisibleRootItem())
            self.list.expanded.connect(lambda: self.list.resizeColumnToContents(0))
            # self.list.expand()
            proxy = self.list.model()

            for row in range(proxy.rowCount()):
                index = proxy.index(row, 0)
                self.list.expand(index)

            self.bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            layout.addWidget(self.bb)
            # layout.addStretch(30)

            self.setLayout(layout)
            self.setMinimumWidth(500)

            self.bb.accepted.connect(self.accept)
            self.bb.rejected.connect(self.reject)
            self.list.resizeColumnToContents(0)

    class Elem:
        class Kind(Enum):
            STR = 1
            INT = 2
            FILE = 3
            CHECKBOX = 4
            FLOAT = 5
            COMBOBOX = 6
            SECTION = 7
            SUBSECTION = 8
            PASSWORD = 9
            EDITBOX = 10
            FILE_SAVE = 11
            CHOSE_DIR = 12
            ROOT = 254

        def __init__(self, key, pretty, kind, value=None, extra=None, parent=None):
            self.kind = kind
            self.key = key
            self.value = value
            self.extra = extra
            self.pretty = pretty
            self.child = []
            self.parent = parent
            self.node = None
            self.w = None
            self._ = self
            self.__ = self
            self.___ = self
            self.____ = self
            self._____ = self
            self.______ = self
            self._______ = self
            self.________ = self
            self._________ = self
            self.___________ = self

        def add(self, key, kind=Kind.STR, default=None, extra=None, pretty=None):
            pretty = pretty if pretty else key
            self.addChild(EasyConfig.Elem(key, pretty, kind, default, extra, parent=self))
            return self

        def addString(self, name, pretty=None, default=None):
            return self.add(name, EasyConfig.Elem.Kind.STR, default, None, pretty)

        def addEditBox(self, name, pretty=None, default=None):
            return self.add(name, EasyConfig.Elem.Kind.EDITBOX, default, None, pretty)

        def addPassword(self, name, pretty=None, default=None):
            return self.add(name, EasyConfig.Elem.Kind.PASSWORD, default, None, pretty)

        def addInt(self, name, pretty=None, default=None):
            return self.add(name, EasyConfig.Elem.Kind.INT, default, None, pretty)

        def addFloat(self, name, pretty=None, default=None):
            return self.add(name, EasyConfig.Elem.Kind.FLOAT, default, None, pretty)

        def addFile(self, name, pretty=None, default=None, extension="txt"):
            return self.add(name, EasyConfig.Elem.Kind.FILE, default, extension.replace(".", ""), pretty)

        def addFileSave(self, name, pretty=None, default=None, extension="txt"):
            return self.add(name, EasyConfig.Elem.Kind.FILE_SAVE, default, extension.replace(".", ""), pretty)

        def addFolderChoice(self, name, pretty=None, default=None):
            return self.add(name, EasyConfig.Elem.Kind.CHOSE_DIR, default, None, pretty)

        def addCheckbox(self, name, pretty=None, default=None):
            return self.add(name, EasyConfig.Elem.Kind.CHECKBOX, default, None, pretty)

        def addCombobox(self, name, pretty=None, items=None):
            return self.add(name, EasyConfig.Elem.Kind.COMBOBOX, items[0], items, pretty)

        def addChild(self, elem):
            self.child.append(elem)

        def addSubSection(self, key, pretty=None):
            pretty = pretty if pretty else key
            elem = EasyConfig.Elem(key, pretty, EasyConfig.Elem.Kind.SUBSECTION, parent=self)
            self.addChild(elem)
            return elem

        def addHidden(self, key):
            elem = EasyConfig.Elem(key, "@", EasyConfig.Elem.Kind.SUBSECTION, parent=self)
            self.addChild(elem)
            return elem

        def create(self, list, node):
            parent = node
            e = self
            if e.kind == EasyConfig.Elem.Kind.INT:
                self.w = EasyConfig.Dialog.Integer(e.key, e.value, e.pretty)
            elif e.kind == EasyConfig.Elem.Kind.FILE:
                self.w = EasyConfig.Dialog.File(e.key, e.value, e.pretty, e.extra)
            elif e.kind == EasyConfig.Elem.Kind.FILE_SAVE:
                self.w = EasyConfig.Dialog.File(e.key, e.value, e.pretty, e.extra, is_save_dialog=True)
            elif e.kind == EasyConfig.Elem.Kind.CHOSE_DIR:
                self.w = EasyConfig.Dialog.File(e.key, e.value, e.pretty, e.extra, is_folder_choice=True)
            elif e.kind == EasyConfig.Elem.Kind.CHECKBOX:
                self.w = EasyConfig.Dialog.Checkbox(e.key, e.value, e.pretty)
            elif e.kind == EasyConfig.Elem.Kind.COMBOBOX:
                self.w = EasyConfig.Dialog.ComboBox(e.key, e.value, e.pretty, e.extra)
            elif e.kind == EasyConfig.Elem.Kind.FLOAT:
                self.w = EasyConfig.Dialog.Float(e.key, e.value, e.pretty)
            elif e.kind == EasyConfig.Elem.Kind.PASSWORD:
                self.w = EasyConfig.Dialog.Password(e.key, e.value, e.pretty)
            elif e.kind == EasyConfig.Elem.Kind.EDITBOX:
                self.w = EasyConfig.Dialog.EditBox(e.key, e.value, e.pretty)
                self.w.set_value(e.value)
            else:
                self.w = EasyConfig.Dialog.String(e.key, e.value, e.pretty)

            child = QTreeWidgetItem()
            child.setText(0, self.pretty)
            parent.addChild(child)
            list.setItemWidget(child, 1, self.w)
            # list.setItemWidget(child, 0, QLabel(self.pretty))

        def getDictionary(self, dic):
            if self.kind == EasyConfig.Elem.Kind.ROOT:
                dic.clear()
                for c in self.child:
                    c.getDictionary(dic)
            elif self.kind == EasyConfig.Elem.Kind.SUBSECTION:
                dic[self.key] = {}
                dic = dic[self.key]
                for c in self.child:
                    c.getDictionary(dic)
            else:
                dic[self.key] = self.value

        def collect(self):
            if self.kind == EasyConfig.Elem.Kind.ROOT:
                for c in self.child:
                    c.collect()
            elif self.kind == EasyConfig.Elem.Kind.SUBSECTION:
                if not self.pretty.startswith("@"):
                    for c in self.child:
                        c.collect()
            elif not self.pretty.startswith("@") and not self.parent.pretty.startswith("@"):
                self.value = self.w.get_value()

        def load(self, dic, keys=None):
            if self.kind == EasyConfig.Elem.Kind.ROOT:
                keys = []
                for c in self.child:
                    c.load(dic, keys.copy())
            elif self.kind == EasyConfig.Elem.Kind.SUBSECTION:
                keys.append(self.key)
                for c in self.child:
                    c.load(dic, keys.copy())
            else:
                for k in keys:
                    dic = dic.get(k)
                    if dic is None:
                        break

                if dic is not None:
                    self.value = dic.get(self.key)

        def trigger(self, list, node=None):
            if self.kind == EasyConfig.Elem.Kind.ROOT:
                node = list.invisibleRootItem()
            elif self.kind == EasyConfig.Elem.Kind.SUBSECTION:
                if not self.pretty.startswith("@"):
                    qtw = QTreeWidgetItem()
                    qtw.setText(0, self.pretty)
                    node.addChild(qtw)
                    node = qtw
            elif not self.pretty.startswith("@") and not self.parent.pretty.startswith("@"):
                self.create(list, node)

            for c in self.child:
                c.trigger(list, node)

    def __init__(self):
        self.root_node = self.Elem("root", "root", EasyConfig.Elem.Kind.ROOT)
        self.reserved = 'main'
        self.expanded = None

    def tab(self):
        return self

    def root(self):
        return self.root_node

    def get_widget(self):
        dialog = self.Dialog(self.root_node)
        dialog.bb.setVisible(False)
        if self.expanded:
            dialog.set_expanded(self.expanded)
        return dialog

    def edit(self):

        dialog = self.Dialog(self.root_node)
        if self.expanded:
            dialog.set_expanded(self.expanded)

        res = dialog.exec()
        self.expanded = dialog.get_expanded()

        if res == 1:
            self.root_node.collect()

    def get(self, key):
        if key is None:
            return None
        elif not key.startswith("/"):
            nodes = self.get_nodes(key)
            if len(nodes) > 0:
                return nodes[0].value
            else:
                raise Exception("Key {} not found".format(key))
        else:
            path = key[1:].split("/")
            node = self.get_node(path)
            if node.key == path[-1]:
                return node.value
            else:
                raise Exception("Key {} not found".format(key))

    def set(self, key, value):
        if key is None:
            return None
        elif not key.startswith("/"):
            nodes = self.get_nodes(key)
            if len(nodes) > 0:
                nodes[0].value = value
                return True
            else:
                raise Exception("Key {} not found".format(key))
        else:
            path = key[1:].split("/")
            node = self.get_node(path)
            if node:
                node.value = value
            return node is not None

    def store_easyconfig_info(self, tree):
        if self.expanded:
            tree["easyconfig"] = {"expanded": ''.join(str(e) for e in self.expanded)}

    def recover_easyconfig_info(self, tree):
        expanded = tree.get("easyconfig", {}).get("expanded")
        if expanded:
            self.expanded = [int(a) for a in expanded]

    def save(self, filename):
        tree = dict()
        self.root_node.getDictionary(tree)
        self.store_easyconfig_info(tree)

        with open(filename, 'w') as f:
            yaml.dump(tree, f, sort_keys=False)

    def get_node(self, keys):
        node = self.root_node
        for p in keys:
            for c in node.child:
                if c.key.lower() == p.lower():  # or c.pretty.lower() == p.lower():
                    node = c
        return node

    def get_nodes(self, key):
        def recu(node, found):
            if node and key and node.kind != EasyConfig.Elem.Kind.SUBSECTION and (node.key.lower() == key.lower()):  # or node.pretty.lower() == key.lower()):
                found.append(node)
            for c in node.child:
                recu(c, found)

        nodes = []
        recu(self.root_node, nodes)
        return nodes

    def load(self, filename):
        try:
            with open(filename, 'r') as f:
                config = yaml.safe_load(f)
                self.recover_easyconfig_info(config)
                self.root_node.load(config)
        except:
            pass


class MainWindow(QPushButton):
    def __init__(self):
        super().__init__()
        self.setText("Try!")
        self.setGeometry(QRect(100, 100, 100, 100))

        self.c = EasyConfig()
        first_level = self.c.root().addSubSection("first_level","First level")
        first_level.addInt("int", "One int")
        first_level.addInt("float", "One float")
        second_level = first_level.addSubSection("second_level","Second Level")
        second_level.addCheckbox("checkbox","The checkbox")
        first_level.addInt("float2", "Another float")
        private = self.c.root().addHidden("private")
        more_private = private.addHidden("more_private")
        more_private.addInt("private_int")

        self.clicked.connect(self.test)

    def test(self):
        self.c.load("co.yaml")
        self.c.set("float", 88)
        self.c.set("private_int", 145)
        print(self.c.get("/private/more_private/private_int"))
        self.c.edit()
        self.c.save("co.yaml")


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()