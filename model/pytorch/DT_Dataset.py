import torch
from torch.utils.data import Dataset

import pandas as pd
import cv2

import sys

ID2WORD = ['\\Im', 'z', '\\mathcal{O}', '\\triangleq', ']', 'p', '\\rightrightarrows', 'J', '\\with', '\\lceil', '0',
           '\\ni', '\\fullmoon', '<', '\\mathfrak{S}', '\\vee', 'F', '\\aa', '\\Pi', '\\bot', '\\int', '\\subset',
           '\\sun', '\\hbar', '\\setminus', '\\square', '\\omega', '\\mathcal{F}', 'y', '\\star', '\\mathds{E}',
           '\\prec', 'c', '\\mathcal{E}', '\\searrow', '\\mathcal{G}', '\\rrbracket', '\\mathbb{1}', '7', '\\ae', '\\o',
           'Q', 'C', '\\mathds{1}', '\\mathcal{Z}', '\\log', '\\circ', '\\mathds{C}', '\\varepsilon', '\\amalg',
           '\\rightharpoonup', '\\circledcirc', '\\Bowtie', '\\sim', '\\Leftarrow', '+', '\\dots', '\\alpha', '\\vdots',
           '\\copyright', '\\mathds{R}', 'l', '\\venus', '-', '\\backsim', '\\doteq', '\\gtrless', 'H', '\\odot',
           '\\mathbb{Z}', '\\diamondsuit', '\\rangle', '\\mathfrak{A}', 'o', '\\blacksquare', '\\prod', '\\%',
           '\\langle', '\\supset', '\\gamma', '\\blacktriangleright', '\\approx', '\\lozenge', '\\mathcal{T}',
           '\\frown', '\\|', '\\epsilon', 'Y', '\\longrightarrow', '2', 'f', 'u', '\\mathscr{D}', '\\sin', '\\varpi',
           '\\vdash', '\\male', '\\in', '\\iddots', '\\preceq', '\\lfloor', 'V', '\\Downarrow', '\\rtimes', '\\varphi',
           '\\mapsfrom', '\\mathcal{C}', '\\iota', '\\phi', '\\mid', '\\notin', '\\lesssim', '\\mathscr{E}', '\\top',
           '[', '\\mathscr{H}', '\\cong', '\\rho', '\\downarrow', '\\rfloor', '\\sqcap', '\\nvDash', '\\supseteq',
           '\\uplus', '\\shortrightarrow', 'q', '\\nexists', '\\mathds{N}', '\\varkappa', '\\succeq', '\\nRightarrow',
           '\\ast', '\\rightleftarrows', '\\circlearrowleft', '\\succ', 'm', 'E', '\\otimes', '\\infty',
           '\\mathfrak{M}', '\\leqslant', '\\preccurlyeq', 'B', 'Z', '\\Sigma', '\\Gamma', '\\vartheta', '\\ominus',
           '\\AA', '\\ohm', '\\rightleftharpoons', '\\asymp', 'U', 'i', '\\dag', '4', '\\mathcal{D}', '\\cos',
           '\\backslash', '\\sqsubseteq', 'd', '\\Vdash', '\\sum', '\\subseteq', '\\kappa', '\\div', 'e', 'G',
           '\\oiint', 'x', '\\checked', '\\between', '\\longmapsto', 'O', '\\diameter', '\\prime', '\\mathds{P}',
           '\\heartsuit', '\\tan', '\\pi', '\\vartriangle', 's', '\\Re', '\\uparrow', '\\bowtie', '\\mathscr{C}',
           '\\xi', '\\lambda', '\\sharp', '\\oplus', '\\mathscr{P}', '\\sqrt{}', '\\mp', '>', '\\pounds',
           '\\upharpoonright', 'g', '\\emptyset', '\\therefore', 'a', '\\sphericalangle', '8', '\\nsubseteq',
           '\\varoiint', '\\pitchfork', '\\varnothing', '5', '\\boxtimes', '\\mathscr{F}', 'j', 'k', '\\rightarrow',
           '\\&', '\\leftarrow', '\\$', 'w', '\\equiv', '\\because', '\\mathcal{M}', '6', '\\leq', '\\nrightarrow',
           '\\mathbb{Q}', '\\L', '\\twoheadrightarrow', '\\rceil', '\\sigma', '\\varpropto', '\\trianglelefteq',
           '\\cup', '\\mathds{Z}', 'v', '\\mathsection', '\\parr', '\\Psi', '\\theta', '/', '\\wr', '\\gtrsim',
           '\\Longleftrightarrow', '\\mathds{Q}', 'P', '\\nmid', '\\mathcal{R}', '\\coprod', '\\triangledown',
           '\\Longrightarrow', '\\}', 'I', '\\lightning', '\\chi', '\\wp', '\\mathcal{A}', 'X', '\\propto', 'n',
           '\\leadsto', '\\diamond', '\\celsius', '\\tau', '\\nu', '\\psi', '\\sqcup', '\\boxdot', '\\barwedge',
           '\\mathfrak{X}', '\\nabla', '\\forall', '\\lim', '\\not\\equiv', '\\mathcal{L}', '\\mathcal{U}', '\\geq',
           '\\mathcal{N}', '\\Leftrightarrow', '\\perp', 'b', '\\Omega', 'T', '\\models', '\\leftmoon', '9', '\\vDash',
           '\\Theta', '\\Phi', '\\mathcal{H}', '\\ss', '\\varrho', '\\checkmark', '\\llbracket', '\\times', '\\flat',
           'M', '\\simeq', '\\S', '\\Rightarrow', '\\mathbb{H}', 'R', '\\neg', 'h', '\\astrosun', '\\geqslant',
           '\\mapsto', '\\varsubsetneq', '\\cdot', '\\pm', '\\ltimes', 'L', '\\triangle', '\\ddots', '\\neq', '\\delta',
           '|', '\\partial', '\\dashv', '\\mathcal{P}', '\\Xi', '\\triangleleft', 'r', '1', '\\AE', '\\lhd',
           '\\mathcal{X}', 'S', '\\mathbb{R}', '\\boxplus', '\\mathscr{S}', '\\O', '\\exists', 'W', '\\circledast',
           '\\subsetneq', '\\eta', 'N', '\\mathcal{S}', '\\female', '\\mu', '\\Delta', '\\hookrightarrow', '\\fint',
           '\\circledR', '\\triangleright', '\\curvearrowright', '\\leftrightarrow', '\\clubsuit', '\\ell', '\\beta',
           '\\degree', '\\{', 'D', '\\mathscr{A}', '\\bullet', 'K', '\\#', '\\aleph', '\\multimap', '\\rightsquigarrow',
           'A', '\\guillemotleft', '\\mathscr{L}', '\\angle', '\\dotsc', '\\mathcal{B}', '\\zeta', '\\cap', '\\oint',
           '\\circlearrowright', '\\mars', '\\mathbb{N}', '\\parallel', '\\nearrow', '3', '\\Lambda', '\\wedge']

WORD2ID = {'\\Im': 0, 'z': 1, '\\mathcal{O}': 2, '\\triangleq': 3, ']': 4, 'p': 5, '\\rightrightarrows': 6, 'J': 7,
           '\\with': 8, '\\lceil': 9, '0': 10, '\\ni': 11, '\\fullmoon': 12, '<': 13, '\\mathfrak{S}': 14, '\\vee': 15,
           'F': 16, '\\aa': 17, '\\Pi': 18, '\\bot': 19, '\\int': 20, '\\subset': 21, '\\sun': 22, '\\hbar': 23,
           '\\setminus': 24, '\\square': 25, '\\omega': 26, '\\mathcal{F}': 27, 'y': 28, '\\star': 29,
           '\\mathds{E}': 30, '\\prec': 31, 'c': 32, '\\mathcal{E}': 33, '\\searrow': 34, '\\mathcal{G}': 35,
           '\\rrbracket': 36, '\\mathbb{1}': 37, '7': 38, '\\ae': 39, '\\o': 40, 'Q': 41, 'C': 42, '\\mathds{1}': 43,
           '\\mathcal{Z}': 44, '\\log': 45, '\\circ': 46, '\\mathds{C}': 47, '\\varepsilon': 48, '\\amalg': 49,
           '\\rightharpoonup': 50, '\\circledcirc': 51, '\\Bowtie': 52, '\\sim': 53, '\\Leftarrow': 54, '+': 55,
           '\\dots': 56, '\\alpha': 57, '\\vdots': 58, '\\copyright': 59, '\\mathds{R}': 60, 'l': 61, '\\venus': 62,
           '-': 63, '\\backsim': 64, '\\doteq': 65, '\\gtrless': 66, 'H': 67, '\\odot': 68, '\\mathbb{Z}': 69,
           '\\diamondsuit': 70, '\\rangle': 71, '\\mathfrak{A}': 72, 'o': 73, '\\blacksquare': 74, '\\prod': 75,
           '\\%': 76, '\\langle': 77, '\\supset': 78, '\\gamma': 79, '\\blacktriangleright': 80, '\\approx': 81,
           '\\lozenge': 82, '\\mathcal{T}': 83, '\\frown': 84, '\\|': 85, '\\epsilon': 86, 'Y': 87,
           '\\longrightarrow': 88, '2': 89, 'f': 90, 'u': 91, '\\mathscr{D}': 92, '\\sin': 93, '\\varpi': 94,
           '\\vdash': 95, '\\male': 96, '\\in': 97, '\\iddots': 98, '\\preceq': 99, '\\lfloor': 100, 'V': 101,
           '\\Downarrow': 102, '\\rtimes': 103, '\\varphi': 104, '\\mapsfrom': 105, '\\mathcal{C}': 106, '\\iota': 107,
           '\\phi': 108, '\\mid': 109, '\\notin': 110, '\\lesssim': 111, '\\mathscr{E}': 112, '\\top': 113, '[': 114,
           '\\mathscr{H}': 115, '\\cong': 116, '\\rho': 117, '\\downarrow': 118, '\\rfloor': 119, '\\sqcap': 120,
           '\\nvDash': 121, '\\supseteq': 122, '\\uplus': 123, '\\shortrightarrow': 124, 'q': 125, '\\nexists': 126,
           '\\mathds{N}': 127, '\\varkappa': 128, '\\succeq': 129, '\\nRightarrow': 130, '\\ast': 131,
           '\\rightleftarrows': 132, '\\circlearrowleft': 133, '\\succ': 134, 'm': 135, 'E': 136, '\\otimes': 137,
           '\\infty': 138, '\\mathfrak{M}': 139, '\\leqslant': 140, '\\preccurlyeq': 141, 'B': 142, 'Z': 143,
           '\\Sigma': 144, '\\Gamma': 145, '\\vartheta': 146, '\\ominus': 147, '\\AA': 148, '\\ohm': 149,
           '\\rightleftharpoons': 150, '\\asymp': 151, 'U': 152, 'i': 153, '\\dag': 154, '4': 155, '\\mathcal{D}': 156,
           '\\cos': 157, '\\backslash': 158, '\\sqsubseteq': 159, 'd': 160, '\\Vdash': 161, '\\sum': 162,
           '\\subseteq': 163, '\\kappa': 164, '\\div': 165, 'e': 166, 'G': 167, '\\oiint': 168, 'x': 169,
           '\\checked': 170, '\\between': 171, '\\longmapsto': 172, 'O': 173, '\\diameter': 174, '\\prime': 175,
           '\\mathds{P}': 176, '\\heartsuit': 177, '\\tan': 178, '\\pi': 179, '\\vartriangle': 180, 's': 181,
           '\\Re': 182, '\\uparrow': 183, '\\bowtie': 184, '\\mathscr{C}': 185, '\\xi': 186, '\\lambda': 187,
           '\\sharp': 188, '\\oplus': 189, '\\mathscr{P}': 190, '\\sqrt{}': 191, '\\mp': 192, '>': 193, '\\pounds': 194,
           '\\upharpoonright': 195, 'g': 196, '\\emptyset': 197, '\\therefore': 198, 'a': 199, '\\sphericalangle': 200,
           '8': 201, '\\nsubseteq': 202, '\\varoiint': 203, '\\pitchfork': 204, '\\varnothing': 205, '5': 206,
           '\\boxtimes': 207, '\\mathscr{F}': 208, 'j': 209, 'k': 210, '\\rightarrow': 211, '\\&': 212,
           '\\leftarrow': 213, '\\$': 214, 'w': 215, '\\equiv': 216, '\\because': 217, '\\mathcal{M}': 218, '6': 219,
           '\\leq': 220, '\\nrightarrow': 221, '\\mathbb{Q}': 222, '\\L': 223, '\\twoheadrightarrow': 224,
           '\\rceil': 225, '\\sigma': 226, '\\varpropto': 227, '\\trianglelefteq': 228, '\\cup': 229,
           '\\mathds{Z}': 230, 'v': 231, '\\mathsection': 232, '\\parr': 233, '\\Psi': 234, '\\theta': 235, '/': 236,
           '\\wr': 237, '\\gtrsim': 238, '\\Longleftrightarrow': 239, '\\mathds{Q}': 240, 'P': 241, '\\nmid': 242,
           '\\mathcal{R}': 243, '\\coprod': 244, '\\triangledown': 245, '\\Longrightarrow': 246, '\\}': 247, 'I': 248,
           '\\lightning': 249, '\\chi': 250, '\\wp': 251, '\\mathcal{A}': 252, 'X': 253, '\\propto': 254, 'n': 255,
           '\\leadsto': 256, '\\diamond': 257, '\\celsius': 258, '\\tau': 259, '\\nu': 260, '\\psi': 261,
           '\\sqcup': 262, '\\boxdot': 263, '\\barwedge': 264, '\\mathfrak{X}': 265, '\\nabla': 266, '\\forall': 267,
           '\\lim': 268, '\\not\\equiv': 269, '\\mathcal{L}': 270, '\\mathcal{U}': 271, '\\geq': 272,
           '\\mathcal{N}': 273, '\\Leftrightarrow': 274, '\\perp': 275, 'b': 276, '\\Omega': 277, 'T': 278,
           '\\models': 279, '\\leftmoon': 280, '9': 281, '\\vDash': 282, '\\Theta': 283, '\\Phi': 284,
           '\\mathcal{H}': 285, '\\ss': 286, '\\varrho': 287, '\\checkmark': 288, '\\llbracket': 289, '\\times': 290,
           '\\flat': 291, 'M': 292, '\\simeq': 293, '\\S': 294, '\\Rightarrow': 295, '\\mathbb{H}': 296, 'R': 297,
           '\\neg': 298, 'h': 299, '\\astrosun': 300, '\\geqslant': 301, '\\mapsto': 302, '\\varsubsetneq': 303,
           '\\cdot': 304, '\\pm': 305, '\\ltimes': 306, 'L': 307, '\\triangle': 308, '\\ddots': 309, '\\neq': 310,
           '\\delta': 311, '|': 312, '\\partial': 313, '\\dashv': 314, '\\mathcal{P}': 315, '\\Xi': 316,
           '\\triangleleft': 317, 'r': 318, '1': 319, '\\AE': 320, '\\lhd': 321, '\\mathcal{X}': 322, 'S': 323,
           '\\mathbb{R}': 324, '\\boxplus': 325, '\\mathscr{S}': 326, '\\O': 327, '\\exists': 328, 'W': 329,
           '\\circledast': 330, '\\subsetneq': 331, '\\eta': 332, 'N': 333, '\\mathcal{S}': 334, '\\female': 335,
           '\\mu': 336, '\\Delta': 337, '\\hookrightarrow': 338, '\\fint': 339, '\\circledR': 340,
           '\\triangleright': 341, '\\curvearrowright': 342, '\\leftrightarrow': 343, '\\clubsuit': 344, '\\ell': 345,
           '\\beta': 346, '\\degree': 347, '\\{': 348, 'D': 349, '\\mathscr{A}': 350, '\\bullet': 351, 'K': 352,
           '\\#': 353, '\\aleph': 354, '\\multimap': 355, '\\rightsquigarrow': 356, 'A': 357, '\\guillemotleft': 358,
           '\\mathscr{L}': 359, '\\angle': 360, '\\dotsc': 361, '\\mathcal{B}': 362, '\\zeta': 363, '\\cap': 364,
           '\\oint': 365, '\\circlearrowright': 366, '\\mars': 367, '\\mathbb{N}': 368, '\\parallel': 369,
           '\\nearrow': 370, '3': 371, '\\Lambda': 372, '\\wedge': 373}


class DT_Dataset(Dataset):
    def __init__(self, path: str, transform=None, train=True):
        """
        :param path: Path to data folder
        :param transform: Transformations to perform on data
        :param train: True for training data, false for test data
        """
        self.path = path
        self.transform = transform
        self.train = train
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            if self.train:
                self.df = pd.read_csv(f"{path}/train.csv")
            else:
                self.df = pd.read_csv(f"{path}/test.csv")
        except IOError:
            print("Invalid path")
            sys.exit(1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        """
        Retrieves item'th element of dataset
        :param item:
        :return:
        """
        cur_row = self.df.iloc[item]
        img_path = cur_row["path"]
        img = self.prepare_img(f"{self.path}/hasy-data/{img_path}")
        label = WORD2ID[cur_row["latex"]]
        label_tensor = torch.tensor(label)

        if self.transform is not None:
            img = self.transform(img)

        return img, label_tensor

    def prepare_img(self, path: str) -> torch.Tensor:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.reshape((1, 45, 45))
        return torch.tensor(img).float()
