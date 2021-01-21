import logging
from collections import defaultdict
import numpy as np
from chmpy.core.element import Element, chemical_formula

LOG = logging.getLogger(__name__)


class AsymmetricUnit:
    """
    Storage class for the coordinates and labels in a crystal
    asymmetric unit

    Attributes:
        elements (List[Element]): N length list of elements associated with the sites in this asymmetric
            unit
        positions (array_like): (N, 3) array of site positions in fractional coordinates
        labels (array_like): N length array of string labels for each site
    """

    def __init__(self, elements, positions, labels=None, **kwargs):
        """
        Create an asymmetric unit object from a list of Elements and
        an array of fractional coordinates.


        Arguments:
            elements (List[Element]): N length list of elements associated with the sites
            positions (array_like): (N, 3) array of site positions in fractional coordinates
            labels (array_like, optional): labels (array_like): N length array of string labels for each site
            **kwargs: Additional properties (will populate the properties member)
                to store in this asymmetric unit

        """
        self.elements = elements
        self.atomic_numbers = np.asarray([x.atomic_number for x in elements])
        self.positions = np.asarray(positions)
        self.properties = {}
        self.properties.update(kwargs)
        if labels is None:
            self.labels = []
            label_index = defaultdict(int)
            for el in self.elements:
                label_index[el] += 1
                self.labels.append("{}{}".format(el, label_index[el]))
        else:
            self.labels = labels
        self.labels = np.array(self.labels)

    @property
    def formula(self):
        """Molecular formula for this asymmetric unit"""
        return chemical_formula(self.elements, subscript=False)

    def __len__(self):
        return len(self.elements)

    def __repr__(self):
        return "<{}>".format(self.formula)

    @classmethod
    def from_records(cls, records):
        """Initialize an AsymmetricUnit from a list of dictionary like objects

        Arguments:
            records (iterable): An iterable containing dict_like objects with `label`,
                `element`, `position` and optionally `occupation` stored.
        """
        labels = []
        elements = []
        positions = []
        occupation = []
        for r in records:
            labels.append(r["label"])
            elements.append(Element[r["element"]])
            positions.append(r["position"])
            occupation.append(r.get("occupation", 1.0))
        positions = np.asarray(positions)
        return AsymmetricUnit(elements, positions, labels=labels, occupation=occupation)
