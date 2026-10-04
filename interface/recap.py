"""Récapitulatif de fin d'analyse : les chiffres clés et la courbe du débit.

Ce module ne calcule AUCUNE mesure nouvelle. Les chiffres affichés sont ceux
que `compteur.rapport.statistiques` produit déjà pour le JSON d'export — le même
total, la même durée, le même débit, le même pic. Les afficher ici à part
inventerait une seconde version de la vérité, et les deux divergeraient dès
qu'un export serait produit sur une autre machine.

La seule chose calculée ici est **la courbe**, c'est-à-dire la répartition des
franchissements dans le temps : l'export donne un total et un pic, pas la
courbe entre les deux. Elle se lit sur les événements, qui sont datés, et sur
rien d'autre.

## Deux unités distinctes, et pourquoi

- **Durée analysée** — durée de la VIDÉO (`duree_video_s`, ou `secondes` en
  repli). C'est elle qui divise le total pour donner un débit par minute.
- **Pic de débit** — nombre d'événements dans une fenêtre glissante de 60 s
  (`debit_max_par_minute`), repris tel quel. Si l'analyse a duré moins d'une
  minute, la fenêtre réelle est plus courte : le pic est alors affiché comme
  un compte sur cette fenêtre (« 12 pers / 40 s ») et non comme un débit par
  minute. Un chiffre par minute sur une fenêtre de 40 s serait une
  extrapolation que rien ne mesure.
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from compteur.rapport import FENETRE_DEBIT_S, statistiques
from compteur.types import Resultat

#: Titre du bloc. Sobre : c'est un outil de travail, pas une application de
#: consommateurs. Aucun « Félicitations », aucune émoticône.
TITRE_RECAP = "Analyse terminée"

LIBELLE_TOTAL = "Personnes comptées"
LIBELLE_DUREE = "Durée analysée"
LIBELLE_DEBIT = "Débit moyen"
LIBELLE_PIC = "Pic de débit"

#: Légende sous la courbe : elle dit ce que l'axe vertical compte, faute de
#: quoi un pic de 147 se lit comme 147 personnes.
LEGENDE_COURBE = "Personnes par minute, en fonction du temps"


# -- Formatage ------------------------------------------------------------


def formater_duree(secondes: float) -> str:
    """Une durée en « 2 min 41 s », lisible d'un coup d'œil.

    Sous une minute, les secondes seules ; au-delà d'une heure, les minutes
    seules — un compteur de personnes n'a pas besoin de la seconde à ce niveau.
    """
    total = max(0, int(round(float(secondes or 0.0))))
    if total >= 3600:
        return f"{total // 3600} h {int(round((total % 3600) / 60)):02d} min"
    if total >= 60:
        return f"{total // 60} min {total % 60:02d} s"
    return f"{total} s"


def formater_debit(valeur: float) -> str:
    """Un débit en « 95 pers/min », à l'unité près."""
    return f"{int(round(float(valeur or 0.0)))} pers/min"


# -- La courbe ------------------------------------------------------------


def duree_analysee(resultat: Resultat) -> float:
    """Durée sur laquelle le débit a été calculé, en secondes.

    Même règle que `rapport.statistiques` : la durée de la vidéo si elle est
    connue, le temps de calcul sinon. Confondre les deux sous-estimerait le
    débit d'un facteur vidéo/calcul.
    """
    video = float(resultat.duree_video_s or 0.0)
    return video if video > 0.0 else float(resultat.secondes or 0.0)


def points_de_debit(resultat: Resultat, fenetre_s: float = FENETRE_DEBIT_S) -> list:
    """Nombre de franchissements par tranche de `fenetre_s` secondes.

    Rend une liste de couples ``(début de la tranche en secondes, nombre)``,
    une tranche par minute de l'analyse, dans l'ordre. Les minutes sans
    franchissement sont présentes et valent 0 : une courbe où les trous sont
    absents lirait « personne ne passe » là où la tranche n'a simplement pas
    été vue.

    Le nombre de tranches est déduit de la durée, pas des événements : une
    analyse de 3 minutes donne trois points même si personne n'a traversé, et
    c'est cette information — le silence — que la courbe doit montrer.
    """
    duree = duree_analysee(resultat)
    if duree <= 0.0 or fenetre_s <= 0.0:
        return []
    nb = max(1, int(math.ceil(duree / fenetre_s)))
    comptes = [0] * nb
    for ev in resultat.evenements:
        t = float(ev.timestamp_s)
        if t < 0.0:
            continue
        index = int(t // fenetre_s)
        # Un événement daté après la durée annoncée (horloge vidéo bizarre,
        # video/calcul confondues) ne doit pas étendre la courbe ni faire
        # disparaître les tranches réelles : il est compté dans la dernière.
        index = min(nb - 1, max(0, index))
        comptes[index] += 1
    return [(i * fenetre_s, comptes[i]) for i in range(nb)]


def instant_du_pic(evenements: list, fenetre: float, valeur: int) -> float | None:
    """Début de la PREMIÈRE fenêtre glissante qui atteint `valeur` événements.

    La fenêtre est la même que celle de `rapport._pic_de_debit`, et le
    décompte aussi : la valeur du pic vient du rapport, cette fonction ne
    fait que dire À QUANDEL moment elle a été observée. Elle ne recalcule donc
    pas le pic — elle le localise.

    `None` si la valeur n'est atteinte par aucune fenêtre (fenêtre nulle,
    événements absents) : un pic sans instant ne doit pas être affiché.
    """
    if not evenements or valeur <= 0:
        return None
    horodatages = sorted(float(ev.timestamp_s) for ev in evenements)
    if fenetre <= 0.0:
        return horodatages[0]
    debut = 0
    for fin, t in enumerate(horodatages):
        while horodatages[debut] < t - fenetre:
            debut += 1
        if fin - debut + 1 == valeur:
            return horodatages[debut]
    return None


class CourbeDebit(QWidget):
    """Tracé du nombre de personnes par minute sur toute l'analyse.

    Le dessin est fait à la main (`paintEvent`) plutôt que par une vue de
    données : un `QChartView` tire QwtPlot et QChart, deux dépendances de plus
    dans un exécutable déjà lourd, pour cinq lignes et un axe.
    """

    MARGE_G, MARGE_D, MARGE_H, MARGE_B = 46, 12, 10, 24

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._points: list = []
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def definir_points(self, points: list) -> None:
        """Fixe la série ``[(temps_s, personnes), …]`` et redessine."""
        self._points = list(points)
        self.update()

    def points(self) -> list:
        """La série affichée — lisible par les tests et par l'export."""
        return list(self._points)

    def paintEvent(self, _evenement) -> None:  # noqa: N802 (API Qt)
        peintre = QPainter(self)
        peintre.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        peintre.fillRect(self.rect(), QColor("#14161a"))

        if not self._points:
            # Une analyse à durée nulle n'a pas de courbe à tracer. Le silence
            # est explicite, jamais un cadre vide qui laisserait croire à un
            # bug d'affichage.
            peintre.setPen(QPen(QColor("#6b7280")))
            peintre.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Aucune donnée à tracer"
            )
            return

        g, d, h, b = self.MARGE_G, self.MARGE_D, self.MARGE_H, self.MARGE_B
        zone = QRectF(
            g,
            h,
            max(1.0, self.width() - g - d),
            max(1.0, self.height() - h - b),
        )

        valeurs = [v for _, v in self._points]
        t_max = max(t for t, _ in self._points) or 1.0
        v_max = max(valeurs)
        # Un maximum nul donnerait une division par zéro : on pose alors un
        # plafond de 1, et la courbe est une ligne plate sur le bas du cadre.
        plafond = float(v_max) if v_max > 0 else 1.0

        # -- grille et axes --
        peintre.setPen(QPen(QColor("#2b2f36"), 1))
        for i in range(3):
            y = zone.bottom() - zone.height() * i / 2.0
            peintre.drawLine(QPointF(zone.left(), y), QPointF(zone.right(), y))
        peintre.drawLine(zone.bottomLeft(), zone.bottomRight())

        peintre.setPen(QPen(QColor("#9aa3af")))
        peintre.drawText(
            QRectF(0, zone.top() - 9, g - 6, 18),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            f"{int(round(v_max))}",
        )
        peintre.drawText(
            QRectF(0, zone.bottom() - 9, g - 6, 18),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            "0",
        )
        peintre.drawText(
            QRectF(zone.left() - 8, zone.bottom() + 3, zone.width() / 2, 18),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop),
            "0",
        )
        peintre.drawText(
            QRectF(zone.center().x(), zone.bottom() + 3, zone.width() / 2, 18),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop),
            formater_duree(t_max + FENETRE_DEBIT_S),
        )

        # -- la courbe --
        pas = zone.width() / max(1, len(self._points))
        positions = [
            QPointF(
                zone.left() + pas * (i + 0.5),
                zone.bottom() - zone.height() * (v / plafond),
            )
            for i, (_, v) in enumerate(self._points)
        ]

        aire = QPainterPath()
        aire.moveTo(positions[0].x(), zone.bottom())
        for point in positions:
            aire.lineTo(point)
        aire.lineTo(positions[-1].x(), zone.bottom())
        aire.closeSubpath()
        peintre.fillPath(aire, QColor(74, 222, 128, 38))

        trace = QPainterPath(positions[0])
        for point in positions[1:]:
            trace.lineTo(point)
        peintre.setPen(QPen(QColor("#4ade80"), 2))
        peintre.drawPath(trace)


class PanneauRecap(QWidget):
    """Le bloc « Analyse terminée » : quatre chiffres et une courbe.

    Masqué tant qu'aucune analyse n'est terminée : il occupe de la place dans
    la colonne de droite, et une place vide au-dessus du compteur ferait croire
    qu'il manque quelque chose.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        racine = QVBoxLayout(self)
        racine.setContentsMargins(0, 8, 0, 0)
        racine.setSpacing(8)

        self.titre = QLabel(TITRE_RECAP)
        self.titre.setObjectName("recap_titre")
        racine.addWidget(self.titre)

        grille = QGridLayout()
        grille.setHorizontalSpacing(10)
        grille.setVerticalSpacing(4)
        self.valeurs: dict[str, QLabel] = {}
        for ligne, cle in enumerate(
            (LIBELLE_TOTAL, LIBELLE_DUREE, LIBELLE_DEBIT, LIBELLE_PIC)
        ):
            nom = QLabel(cle)
            nom.setObjectName("recap_libelle")
            nom.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            valeur = QLabel("—")
            valeur.setObjectName("recap_valeur")
            valeur.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            valeur.setWordWrap(True)
            grille.addWidget(nom, ligne, 0)
            grille.addWidget(valeur, ligne, 1)
            self.valeurs[cle] = valeur
        grille.setColumnStretch(1, 1)
        racine.addLayout(grille)

        self.courbe = CourbeDebit()
        racine.addWidget(self.courbe, stretch=1)

        self.legende = QLabel(LEGENDE_COURBE)
        self.legende.setObjectName("sous_titre")
        self.legende.setWordWrap(True)
        racine.addWidget(self.legende)

    # -- Remplissage ----------------------------------------------------

    def afficher(self, resultat: Resultat) -> None:
        """Remplit les chiffres et la courbe à partir du `Resultat` brut.

        Les chiffres viennent de `rapport.statistiques` : l'écran et le JSON
        d'export disent donc la même chose, calculés une seule fois.
        """
        stats = statistiques(resultat)
        duree = float(stats["duree_s"] or 0.0)
        fenetre = float(stats["debit_max_fenetre_s"] or 0.0)
        pic = int(stats["debit_max_par_minute"] or 0)

        self.valeurs[LIBELLE_TOTAL].setText(f"{int(stats['total'] or 0)}")
        self.valeurs[LIBELLE_DUREE].setText(formater_duree(duree))
        self.valeurs[LIBELLE_DEBIT].setText(
            formater_debit(stats["personnes_par_minute"])
            if duree > 0.0
            else "non mesuré"
        )
        self.valeurs[LIBELLE_PIC].setText(self._texte_pic(resultat, pic, fenetre))
        self.courbe.definir_points(points_de_debit(resultat))

    def _texte_pic(self, resultat: Resultat, pic: int, fenetre: float) -> str:
        """« 147 pers/min à 1 min 12 s », ou un compte si la fenêtre est courte."""
        if pic <= 0:
            return "aucun passage compté"
        instant = instant_du_pic(resultat.evenements, fenetre, pic)
        ou = f" à {formater_duree(instant)}" if instant is not None else ""
        if fenetre >= FENETRE_DEBIT_S - 0.5:
            return f"{formater_debit(pic)}{ou}"
        # Fenêtre réelle plus courte qu'une minute : annoncer « pers/min »
        # serait extrapoler sur une minute qui n'a pas été observée.
        return f"{pic} pers / {int(round(fenetre))} s{ou}"

    def effacer(self) -> None:
        """Remet le panneau à zéro et masque la courbe.

        Appelé au lancement d'une nouvelle analyse : un récapitulatif de la
        précédente resterait affiché pendant qu'un nouveau décompte démarre,
        et l'opérateur lirait l'ancien total comme le nouveau.
        """
        for valeur in self.valeurs.values():
            valeur.setText("—")
        self.courbe.definir_points([])