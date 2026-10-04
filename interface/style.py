"""Thème sombre et lisible (usage en extérieur).

Contre-intuitif mais délibéré : l'interface est vue dehors, souvent en plein
jour, parfois sur l'écran d'un portable posé sur un trotoir. Un thème sombre
n'est pas choisi pour l'esthétique, mais parce qu'il est le seul moyen
d'obtenir un chiffre LISIBLE à trois mètres. Sur fond clair, un gros chiffre
noir en plein soleil est illisible (le noir se confond avec le bitume, et
l'œil accommode mal entre une zone claire et une zone sombre voisine) ;
inversé — chiffres clairs sur fond sombre — le contraste reste lisible quelle
que soit la lumière ambiante.

D'où les deux décisions qui structurent cette feuille de style :

- `QLabel#compteur` est en très grand (76 px) et gras. C'est le seul chiffre
  qui doit être lu de loin ; tout le reste peut l'être de près.
- L'accent est un vert vif (#4ade80) sur fond presque noir (#14161a). Un
  ratio de contraste d'environ 13:1, largement au-dessus du seuil de 4.5:1
  des recommandations WCAG AA, et il reste lisible en plein jour.

`appliquer_style(app)` est idempotent : l'appeler deux fois donne le même
résultat.
"""

from __future__ import annotations

# Feuilles de style Qt. Ce sont des quasi-CSS : les sélecteurs s'appliquent
# par nom de classe Qt (`QPushButton`) ou par `objectName` (`#compteur`), que
# l'IHM fixe via `setObjectName`.
#
# Les couleurs sont en RVB, comme le reste du projet (OpenCV travaille en BGR,
# mais Qt n'a rien à voir avec la frame).
STYLE = """
QWidget { background-color: #14161a; color: #e6e6e6; font-size: 13px; }
QPushButton {
    background-color: #262a31; border: 1px solid #3a4048; border-radius: 5px;
    padding: 7px 14px;
}
QPushButton:hover { background-color: #31363f; }
QPushButton:disabled { color: #6b7280; border-color: #2a2e35; }
QPushButton#primaire { background-color: #2563eb; border-color: #2563eb; color: #fff; }
QPushButton#primaire:hover { background-color: #1d4ed8; }
QGroupBox {
    border: 1px solid #2b2f36; border-radius: 6px; margin-top: 14px; padding-top: 8px;
}
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #9aa3af; }
QSlider::groove:horizontal { height: 5px; background: #2b2f36; border-radius: 2px; }
QSlider::handle:horizontal {
    background: #2563eb; width: 15px; margin: -6px 0; border-radius: 7px;
}
QLabel#compteur { font-size: 76px; font-weight: 700; color: #4ade80; }
QLabel#compteur_titre { font-size: 14px; color: #9aa3af; letter-spacing: 2px; }
QLabel#sous_titre { color: #9ca3af; }
/* Rappel « 0 compté : vérifie le sens ». Ambre, pas rouge : c'est un
   conseil, pas une erreur, et du rouge concurrencerait le flash de comptage
   et le bouton primaire. Gras pour sauter aux yeux sans être une alerte. */
QLabel#avertissement {
    color: #fbbf24; font-weight: 600; padding-top: 4px;
}
/* Aide sous un reglage : volontairement en petit et gris. Elle est lue de
   pres, jamais de loin — le seul chiffre lu de loin reste le compteur. Elle ne
   doit pas rivaliser avec le champ qu'elle decrit. Elle n'est visible que
   lorsque le bouton « Aide » est actionne (tache 18). */
QLabel#aide_reglage {
    color: #8b929c; font-size: 11px; padding-bottom: 4px;
}
/* Le bouton « Aide » : gris et discret, parce que replier la documentation est
   l'action de fond d'écran, pas le bouton principal de l'écran — « Lancer »
   garde cette place. */
QPushButton#bouton_aide {
    color: #9ca3af; text-align: left; padding: 4px 6px;
}
QPushButton#bouton_aide:checked { color: #e5e7eb; }
/* Titre d'un réglage situé HORS du panneau (la vitesse de présentation, sous
   la barre de boutons). Même aspect que les autres titres, pour que le
   curseur ne paraisse pas foreign. */
QLabel#libelle_reglage { color: #9aa3af; }
QProgressBar {
    border: 1px solid #2b2f36; border-radius: 4px; text-align: center; height: 18px;
}
QProgressBar::chunk { background-color: #2563eb; }
"""


def appliquer_style(app) -> None:
    """Applique le thème sombre à l'application Qt.

    À appeler une fois, après la création de la QApplication et avant la
    construction des widgets. Le paramètre est laissé non typé (`Any`) : ce
    module n'importe volontairement pas PySide6, et reste donc utilisable
    sans dépendance installée (les tests n'ont pas besoin de l'appeler).
    """
    app.setStyleSheet(STYLE)