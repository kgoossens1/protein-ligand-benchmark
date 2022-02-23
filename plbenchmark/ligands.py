"""
ligands.py
Functions and classes for handling the ligand data.
"""

import os
import re
import io
import yaml
import pandas as pd

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, PandasTools
from openff.toolkit.topology import Molecule

from . import targets, utils


class Ligand:
    """
    Store and convert the data of one ligand in a :py:class:`pandas.Series`.

    """

    def __init__(self, d: dict, target: str = None):
        """
        Initialize :py:class:`plbenchmark.ligands.ligand` object from :py:class:`dict` and store data in a
        :py:class:`pandas.Series`.

        :param d: :py:class:`dict` with the ligand data
        :return None
        """
        self._target = target
        self._data = pd.Series(d)
        self._molecule = None
        try:
            self._name = self._data["name"]
        except KeyError:
            print("no name found in data.")

    def get_name(self):
        """
        Access the name of the ligand.

        :return: name: string
        """
        return self._data["name"][0]

    def get_dataframe(self, columns=None):
        """
        Access the ligand data as a :py:class:`pandas.Dataframe`

        :param columns: list of columns which should be returned in the :py:class:`pandas.Dataframe`
        :return: :py:class:`pandas.Dataframe`
        """
        if columns:
            return self._data[columns]
        else:
            return self._data

    def find_links(self):
        """
        Processes primary data to have links in the html string of the ligand data

        :return: None
        """
        if ("measurement", "doi") in list(self._data.index):
            doi = self._data["measurement", "doi"]
            result = []
            if str(doi) != "nan":
                for ddoi in re.split(r"[; ]+", str(doi)):
                    result.append(utils.find_doi_url(ddoi))
            self._data["measurement", "doi_html"] = r"\n".join(result)
            self._data.drop([("measurement", "doi")], inplace=True)
            self._data.rename({"doi_html": "Reference"}, level=1, inplace=True)

    def get_coordinate_file_path(self):
        """
        Get file path relative to the plbenchmark repository of the SDF coordinate file of the docked ligand

        :return: file path as string
        """
        filename = os.path.abspath(
            os.path.join(
                targets.data_path,
                targets.get_target_dir(self._target),
                "02_ligands",
                self._name,
                "crd",
                f"{self._name}.sdf",
            )
        )
        return filename

    def get_molecule(self):
        """
        Get molecule object with coordinates of the docked ligand

        :return: file path as string
        """
        if self._molecule is None:
            filename = self.get_coordinate_file_path()
            self._molecule = Molecule.from_file(filename, "sdf")
        return self._molecule

    def add_mol_to_frame(self):
        """
        Adds a image file of the ligand to the :py:class:`pandas.Dataframe`

        :return: None
        """
        PandasTools.AddMoleculeColumnToFrame(
            self._data, smilesCol="smiles", molCol="ROMol", includeFingerprints=False
        )
        self._data["ROMol"].apply(lambda x: x[0])

    def get_html(self, columns=None):
        """
        Access the ligand as a HTML string

        :param columns: list of columns which should be returned in the :py:class:`pandas.Dataframe`
        :return: HTML string
        """
        self.find_links()
        if columns:
            html_string = pd.DataFrame(self._data[columns]).to_html()
        else:
            html_string = pd.DataFrame(self._data).to_html()
        html_string = html_string.replace("REP1", '<a target="_blank" href="')
        html_string = html_string.replace("REP2", '">')
        html_string = html_string.replace("REP3", "</a>")
        html_string = html_string.replace("\\n", "<br>")
        return html_string

    def get_image(self):
        """
        Creates a molecule image.

        :return: :py:class:`PIL.Image` object
        """
        mol_drawing = Draw.MolDraw2DCairo(200, 200)
        opts = mol_drawing.drawOptions()
        opts.clearBackground = True
        mol = Chem.MolFromSmiles(self._data["smiles"][0])
        Chem.rdDepictor.Compute2DCoords(mol)
        mol_drawing.DrawMolecule(mol, legend=self._data["name"][0])

        # change to transparent background
        img = Image.open(io.BytesIO(mol_drawing.GetDrawingText())).convert("RGBA")
        data = img.getdata()
        new_data = []
        for item in data:
            if item[0] == 255 and item[1] == 255 and item[2] == 255:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)
        img.putdata(new_data)

        return img


class LigandSet(dict):
    """
    Class inherited from dict to store all available ligands of one target.

    """

    def __init__(self, target, *arg, **kw):
        """
        Initializes :py:class:`~plbenchmark.ligands.ligandSet` class

        :param target: string name of target
        :param arg: arguments for :py:class:`dict` (base class)
        :param kw: keywords for :py:class:`dict` (base class)
        """
        super(LigandSet, self).__init__(*arg, **kw)
        target_path = targets.get_target_data_path(target)
        file = open(os.path.join(target_path, "ligands.yml"))
        data = yaml.full_load(file)
        for name, d in data.items():
            lig = Ligand(d, target)
            self[name] = lig
        file.close()

    def get_list(self):
        """
        Returns list of ligands

        :return: :py:class:`list` of ligand names
        """
        return list(self.keys())

    def get_ligand(self, name):
        """
        Accesses one ligand of the :py:class:`~plbenchmark:ligands.ligandSet`

        :param name: string name of the ligand
        :return: :py:class:`plbenchmark.ligands.ligand` class
        """
        if name in self:
            return self[name]
        else:
            raise ValueError(f"Ligand {name} is not part of set.")

    def get_dataframe(self, columns=None):
        """
        Access the :py:class:`~plbenchmark:ligands.ligandSet` as a :py:class:`pandas.Dataframe`

        :param columns: :py:class:`list` of columns which should be returned in the :py:class:`pandas.Dataframe`
        :return: :py:class:`pandas.Dataframe`
        """
        dfs = []
        for key, item in self.items():
            dfs.append(item.get_dataframe(columns))
        df = pd.concat(dfs, axis=1).T
        return df

    def get_html(self, columns=None):
        """
        Access the :py:class:`plbenchmark:ligands.ligandSet` as a HTML string

        :param columns: :py:class:`list` of columns which should be returned in the :py:class:`pandas.Dataframe`
        :return: HTML string
        """
        for key, item in self.items():
            item.find_links()
        df = self.get_dataframe(columns)
        html_string = df.to_html()
        html_string = html_string.replace("REP1", '<a target="_blank" href="')
        html_string = html_string.replace("REP2", '">')
        html_string = html_string.replace("REP3", "</a>")
        html_string = html_string.replace("\\n", "<br>")
        return html_string

    def get_molecules(self):
        """
        Returns a :py:class:`dict` with names as keys and values as py:class:`openforcefield:topology:Molecule` objects

        :return:  :py:class:`dict`
        """
        molecules = {}
        for key, item in self.items():
            molecules[key] = item.get_molecule()
        return molecules
