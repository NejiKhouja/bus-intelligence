"""Traduction minimale à base de clés pour le dashboard Streamlit.

Pas de librairie externe (gettext/babel) -- juste un dict {langue: {clé: gabarit}} et un
sélecteur stocké dans st.session_state["lang"]. `t(key, **kwargs)` cherche le gabarit dans la
langue active, retombe sur le français puis sur la clé brute si absent des deux -- une page
partiellement traduite ne casse jamais, elle affiche juste du français ou la clé en attendant
sa traduction.

Portée actuelle : la page "Détection d'anomalies" (le plus gros morceau du dashboard, et celui
qui a été retravaillé cette session). Les autres pages (Tableau de bord, ETA en direct, Repli
GPS, Assistant, Prévisions) ne sont PAS encore couvertes -- ajouter leurs clés ici au fur et à
mesure suit exactement le même schéma.
"""
import streamlit as st

LANGS = {"fr": "Français", "en": "English", "ar": "العربية"}
# Langues à écriture de droite à gauche -- Streamlit ne retourne pas la mise en page
# automatiquement pour ces langues (colonnes/alignement restent LTR), seul le TEXTE
# s'affiche en arabe. `st.markdown`/`st.caption` avec `unsafe_allow_html=True` peuvent
# forcer `dir="rtl"` localement si besoin plus tard -- non fait ici pour rester simple.
RTL_LANGS = {"ar"}

TRANSLATIONS: dict[str, dict[str, str]] = {
    "fr": {
        # Navigation / structure
        "page_title_anomaly": "Détection d'anomalies",
        "page_subtitle_anomaly": "Trajets signalés par Isolation Forest + autoencodeur LSTM — expliqués en langage clair.",
        "tab_live": "Trajets signalés",
        "tab_explain": "Expliquer un bus",
        "tab_patterns": "Tendances",
        "tab_tickets": "Anomalies billetterie",
        "lang_label": "Langue",

        # Filtres communs
        "filter_operator": "Opérateur",
        "filter_line": "Ligne",
        "filter_direction": "Direction",
        "filter_both_directions": "Les deux",
        "filter_all_lines": "Toutes les lignes",
        "filter_bus": "Bus",
        "filter_all_buses": "Tous les bus",
        "filter_day": "Jour",
        "filter_all_days": "Tous les jours",
        "filter_anomaly_type": "Filtrer par type d'anomalie",
        "sort_by": "Trier par",
        "sort_date_desc": "📅 Date (plus récent d'abord)",
        "sort_date_asc": "📅 Date (plus ancien d'abord)",
        "sort_severity_desc": "🚨 Gravité (élevée d'abord)",
        "sort_severity_asc": "🚨 Gravité (faible d'abord)",
        "sort_duration_desc": "⏱️ Durée (plus longue d'abord)",
        "btn_analyze": "Analyser",
        "show_data_bugs": "Afficher aussi les bugs de données / fragments de suivi",

        # Avertissement modèle à faible historique (voir render_alert_cards) -- affiché une
        # fois par liste quand l'opérateur affiché n'a pas (encore) assez de trajets pour un
        # modèle dédié et retombe sur le modèle GLOBAL (toutes sociétés confondues).
        "model_warning_neither": (
            "Cet opérateur n'a pas encore assez de trajets enregistrés pour un modèle "
            "d'anomalie dédié (ni Isolation Forest, ni autoencodeur LSTM). La détection "
            "ci-dessous compare donc ces trajets à l'ensemble du réseau (toutes sociétés et "
            "lignes confondues), **pas spécifiquement à cette ligne ni à cet opérateur** -- "
            "le résultat n'est pas garanti à 100% et doit être interprété avec prudence. "
            "La précision s'améliorera automatiquement dès que cet opérateur accumulera "
            "plus de données lors des prochaines mises à jour."
        ),
        "model_warning_lstm_only": (
            "Cet opérateur a assez de trajets pour son propre modèle Isolation Forest, mais "
            "pas encore assez (moins de 200 trajets) pour un autoencodeur LSTM dédié -- "
            "celui-ci retombe sur un modèle **global** entraîné sur toutes les sociétés "
            "confondues. Et même l'Isolation Forest \"dédié\" reste entraîné sur TOUTES les "
            "lignes de cet opérateur poolées ensemble, pas sur cette ligne spécifiquement. Le "
            "résultat n'est pas garanti à 100% et doit être interprété avec prudence -- la "
            "précision s'améliorera automatiquement avec plus de données lors des prochaines "
            "mises à jour."
        ),

        # Anomalies billetterie -- vue admin/client (voir main.py::_ticket_rows_with_reasons)
        "tk_view_label": "Vue",
        "tk_view_admin": "Admin (tout)",
        "tk_view_client": "Client (fiable uniquement)",
        "tk_machine_issue": (
            "Panne probable de la machine à tickets : le bus a bien circulé ce jour-là "
            "({trips} trajet(s) GPS confirmé(s)) mais quasi aucun ticket n'a été enregistré. "
            "Ce n'est presque certainement pas une anomalie de recette/fraude -- masqué en "
            "vue client."
        ),
        "tk_no_service": (
            "Aucun trajet GPS ce jour-là pour ce bus -- jour sans service (férié, grève, bus "
            "hors service), pas une anomalie de recette. Masqué en vue client."
        ),
        "tk_good_anomaly": (
            "Bonne anomalie : la recette de ce jour dépasse la normale (de ce bus, ou de la "
            "ligne si pas assez d'historique pour ce bus) -- le modèle la signale car "
            "statistiquement inhabituelle, mais plus d'argent rentré que d'habitude n'est pas "
            "un problème à traiter."
        ),
        "tk_bad_anomaly": (
            "Anomalie à surveiller : la recette de ce jour est en dessous de la normale (de ce "
            "bus, ou de la ligne si pas assez d'historique pour ce bus)."
        ),
        "tk_machine_detection_explain": (
            "Comment on distingue une panne de machine d'une vraie anomalie : chaque jour de "
            "billetterie signalé est croisé avec les trajets GPS réels de CE bus ce jour-là. "
            "Si le GPS confirme que le bus a bien circulé mais que quasi aucun ticket (2 ou "
            "moins) n'a été enregistré, c'est presque certainement que la machine n'a pas "
            "fonctionné -- pas une anomalie de recette. Si le GPS ne montre AUCUN trajet ce "
            "jour-là, c'est un jour sans service (férié, grève, bus hors service), pas une "
            "anomalie non plus. Dans les deux cas : exclu automatiquement en vue client, "
            "affiché avec cette explication en vue admin."
        ),
        "tk_filter_label": "Priorité",
        "tk_filter_bad": "📉 À surveiller (recette basse)",
        "tk_filter_good": "📈 Bonnes (recette haute)",
        "tk_filter_all": "Toutes",

        # En-têtes de section
        "section_flagged_today": "Trajets signalés ce jour",
        "section_recent_history": "Historique récent",
        "metric_operating_day": "Jour d'exploitation",
        "live_data_badge": "Données EN DIRECT (web service GPS, aujourd'hui même)",
        "historical_data_badge": "Données historiques (web service indisponible/pas prêt -- dernier jour connu du dataset)",
        "metric_trips_today": "Trajets ce jour",
        "metric_flagged": "Signalés",
        "no_anomaly_today": "Aucune anomalie le dernier jour d'exploitation pour ce périmètre.",
        "try_specific_line": "Essayez une ligne précise, ou consultez l'historique ci-dessous.",
        "no_history": "Aucune anomalie historique pour ce périmètre.",

        # Carte anomalie -- métriques
        "metric_trip_duration": "Durée trajet",
        "metric_departure_arrival": "Départ → Arrivée",
        "metric_verifiable_activity": "Activité GPS vérifiable",
        "metric_severity": "Gravité",
        "sev_high": "Élevée",
        "sev_medium": "Moyenne",
        "sev_low": "Faible",
        "main_cause": "Cause principale : **{label}**",
        "other_uncategorized": "Autre / non catégorisé",
        "flagged_no_reason": "Signalé par le score du modèle (pas de cause unique dominante).",

        # Libellés courts par feature dominante (filtre multiselect -- texte brut, pas d'icône)
        "topfeat_max_dwell_s": "Immobilisation anormale",
        "topfeat_total_elapsed": "Trajet anormalement long",
        "topfeat_mean_dwell_s": "Durée d'arrêt moyenne élevée",
        "topfeat_dist_m_max": "Déviation de l'itinéraire",
        "topfeat_match_rate": "Mauvais suivi GPS / hors itinéraire",
        "topfeat_n_stops": "Arrêts desservis anormalement faible",
        "topfeat_max_dark_s": "Perte de signal GPS",
        "topfeat_terminus_idle_min": "Stationnement terminus (service non clôturé)",
        "topfeat_elapsed_vs_bus_z": "Durée inhabituelle pour ce bus",
        "topfeat_elapsed_vs_line_z": "Durée inhabituelle pour cette ligne",

        # Formule / activité GPS vérifiable
        "formula_caption": (
            "**Formule** : Durée trajet = heure d'arrivée ({arr}) − heure de départ ({dep}), "
            "**après** avoir retiré {idle:.0f} min de stationnement immobile en bordure du "
            "trajet (avant le vrai départ / après la vraie arrivée). Elle **n'exclut PAS** les "
            "arrêts ou immobilisations survenus EN COURS de trajet (ceux-ci font partie du "
            "service réel et restent comptés dans la durée)."
        ),
        "formula_help": (
            "Durée trajet = dernier ping − premier ping, APRÈS avoir retiré le stationnement "
            "immobile en bordure du trajet (bus garé avant le vrai départ / après la vraie "
            "arrivée). Un arrêt ou une immobilisation EN COURS de trajet, elle, reste comptée "
            "dans cette durée -- ce n'est pas la même chose."
        ),
        "verifiable_activity_caption": (
            "**Activité GPS vérifiable ≈ {est}**  ({dur} − {dwell:.0f} min immobile − "
            "{dark:.0f} min sans signal), avec {match:.0f}% des arrêts suivis. Estimation "
            "basse : le temps de conduite pendant les trous de signal et le stationnement hors "
            "des arrêts reconnus ne sont pas décomptables."
        ),

        # Boîtes qualité (bug/fragment/dark/implausible/partial)
        "q_data_bug": (
            "**Bug de données** : durée > 24h physiquement impossible — horodatages corrompus "
            "dans la source GPS (pas un vrai trajet). Sera éliminé définitivement au prochain "
            "rebuild."
        ),
        "q_fragment": (
            "**Fragment de suivi** : durée très inférieure à la normale de la ligne avec presque "
            "aucun arrêt suivi — trajet partiel ou couverture GPS lacunaire, pas comparable à un "
            "trajet complet."
        ),
        "q_dark_inflated": (
            "**Pourquoi cette durée ?** L'essentiel de cette « durée » est un trou de signal GPS "
            ": le boîtier s'est tu en cours de route puis a réémis bien plus tard. Le bus a très "
            "probablement terminé son service normalement pendant ce silence — la durée affichée "
            "reflète le comportement du boîtier, pas le temps de conduite réel. Ce n'est pas une "
            "erreur du modèle d'anomalie : il signale correctement un suivi GPS défaillant, à "
            "traiter comme tel."
        ),
        "q_implausible": (
            "**Pourquoi cette durée ?** Le boîtier GPS a continué d'émettre après la fin du "
            "service (chauffeur n'ayant pas clôturé le service / bus garé au terminus ou au "
            "dépôt) — ce temps de stationnement s'est fondu dans le trajet. Le vrai temps de "
            "conduite est nettement plus court. Ce n'est pas un défaut du modèle ni de la "
            "segmentation : c'est une pratique d'exploitation (service non clôturé), corrigée "
            "automatiquement au prochain rebuild des données."
        ),
        "q_partial_coverage": (
            "**Pourquoi cette durée ?** Ce trajet n'a couvert que **{ns} arrêt(s)** contre "
            "**{mns:.0f} habituellement** sur cette ligne/direction — le bus a réellement "
            "parcouru une distance plus courte, donc une durée plus courte est normale pour CE "
            "trajet précis. Ce n'est pas une anomalie de vitesse : le comparer à la médiane des "
            "trajets complets serait injuste, comme comparer un aller simple à un aller-retour."
        ),

        # Explications au survol (raisons du modèle, par feature)
        "exp_max_dwell_s": (
            "Le plus long arrêt immobile détecté sur ce trajet dépasse nettement la normale. "
            "Mesuré pings-GPS-présents (bus vraiment immobile), pas un trou de signal."
        ),
        "exp_total_elapsed": (
            "Durée totale du trajet (arrivée − départ), après retrait du stationnement en "
            "bordure de trajet (voir Formule) -- donc déjà nettoyée du service-non-clôturé aux "
            "deux bouts, mais PAS des arrêts survenus en cours de route."
        ),
        "exp_mean_dwell_s": (
            "Durée moyenne d'immobilisation par arrêt sur ce trajet, plus élevée que la normale "
            "de la ligne -- suggère un ralentissement généralisé (trafic, retards en cascade) "
            "plutôt qu'un incident ponctuel à un seul arrêt."
        ),
        "exp_dist_m_max": (
            "Écart maximal observé entre la position GPS du bus et la position théorique d'un "
            "arrêt qu'il a quand même été compté comme ayant desservi -- déviation d'itinéraire "
            "ou dérive GPS."
        ),
        "exp_match_rate": (
            "Part des arrêts de la ligne effectivement détectés par le GPS sur ce trajet. Un "
            "taux bas peut venir d'un trajet partiel, d'un mauvais suivi GPS, ou d'arrêts aux "
            "coordonnées douteuses (voir puces ci-dessous)."
        ),
        "exp_n_stops": (
            "Nombre d'arrêts couverts par ce trajet, nettement inférieur à la normale de la "
            "ligne/direction -- indique un trajet partiel plutôt qu'un problème de vitesse."
        ),
        "exp_max_dark_s": (
            "Le plus long trou de signal GPS détecté (aucun ping reçu) sur ce trajet. Le bus a "
            "pu continuer à rouler normalement pendant ce silence -- ce temps N'EST PAS compté "
            "comme immobilisation, seulement comme incertitude."
        ),
        "exp_terminus_idle_min": (
            "Temps où le traceur GPS a continué à pinger alors que le bus était garé au terminus "
            "-- avant le vrai départ ou après la vraie arrivée. Ce temps est DÉJÀ RETIRÉ de la "
            "durée du trajet affichée (voir Formule) ; ce chiffre le montre séparément."
        ),
        "exp_elapsed_vs_bus_z": (
            "Écart-type (z-score) de la durée de CE trajet par rapport à l'historique de CE BUS "
            "précis sur cette ligne. La durée comparée exclut déjà le stationnement en bordure "
            "de trajet (voir Formule) -- ce n'est pas un artefact de stationnement non retiré."
        ),
        "exp_elapsed_vs_line_z": (
            "Écart-type (z-score) de la durée de CE trajet par rapport à la médiane de TOUS les "
            "bus sur cette ligne/direction. La durée comparée exclut déjà le stationnement en "
            "bordure de trajet (voir Formule) -- ce n'est pas un artefact de stationnement non "
            "retiré."
        ),
        "exp_default": "Signal utilisé par le modèle de détection pour juger ce trajet anormal.",

        # Puces (chips) -- explication au survol
        "chip_terminus_idle": (
            "{icon} Stationnement au terminus : **{min:.0f} min** moteur/traceur actif avant "
            "départ ou après arrivée (service probablement non clôturé — non compté dans la "
            "durée du trajet ci-dessus)."
        ),
        "chip_terminus_idle_help": (
            "Ce temps est RETIRÉ de la « Durée trajet » affichée en haut de la carte -- le bus "
            "pingait encore mais n'avait pas vraiment démarré/terminé son service. Signal "
            "opérationnel à part entière (chauffeur n'ayant probablement pas coupé le traceur), "
            "pas une erreur de mesure."
        ),
        # Variantes nommées (arrêt + horodatage réel) -- utilisées quand ces données sont
        # disponibles (voir foundation.segment_trips) ; `chip_terminus_idle` ci-dessus reste
        # le repli pour les anciens trajets sans ce détail.
        "chip_origin_idle": (
            "{icon} Stationnement au terminus **{stop}** avant le départ : **{min:.0f} min** "
            "— immobile de {from_t} à {to_t}, départ effectif à {to_t} (service probablement "
            "non clôturé — non compté dans la durée du trajet ci-dessus)."
        ),
        "chip_origin_idle_help": (
            "Ce temps est RETIRÉ de la « Durée trajet » affichée en haut de la carte -- le bus "
            "pingait déjà au terminus DE DÉPART mais n'avait pas encore démarré son service. "
            "Signal opérationnel à part entière (chauffeur n'ayant probablement pas coupé le "
            "traceur), pas une erreur de mesure."
        ),
        "chip_end_idle": (
            "{icon} Stationnement au terminus **{stop}** après l'arrivée : **{min:.0f} min** "
            "— immobile de {from_t} à {to_t} (service probablement non clôturé — non compté "
            "dans la durée du trajet ci-dessus)."
        ),
        "chip_end_idle_help": (
            "Ce temps est RETIRÉ de la « Durée trajet » affichée en haut de la carte -- le bus "
            "pingait encore au terminus D'ARRIVÉE mais n'avait pas vraiment terminé son "
            "service. Signal opérationnel à part entière (chauffeur n'ayant probablement pas "
            "coupé le traceur), pas une erreur de mesure."
        ),
        "chip_real_stop": (
            "{icon} Immobilisation réelle : **{stop}** ({min:.0f} min sans mouvement GPS)."
        ),
        "chip_real_stop_help": (
            "Immobilisation détectée EN COURS de trajet (pings GPS présents, bus vraiment "
            "immobile) -- reste comptée dans la « Durée trajet » ci-dessus, contrairement au "
            "stationnement terminus qui lui en est retiré."
        ),
        "chip_detour_hint": (
            "&nbsp;&nbsp;↳ *Si ce délai suit le stationnement terminus ci-dessus, ce n'est pas "
            "forcément le même arrêt : le bus a pu repartir puis revenir avant de s'immobiliser "
            "à nouveau (détour non officiel / course non planifiée). « Voir la carte du trajet » "
            "vérifie sur les pings GPS réels et affiche le trajet emprunté si c'est le cas.*"
        ),
        "chip_same_terminus_hint": (
            "&nbsp;&nbsp;↳ *C'est le MÊME arrêt que le stationnement terminus ci-dessus "
            "(**{stop}**) -- très probablement UNE seule immobilisation continue coupée en deux "
            "par un bref sursaut GPS isolé, pas deux événements distincts. Durée réelle probable : "
            "~{total:.0f} min.*"
        ),
        "chip_signal_loss": "{icon} Perte de signal : **{stop}** (~{min:.0f} min sans ping).",
        "chip_signal_loss_help": (
            "Aucun ping GPS reçu pendant cette durée à cet arrêt -- le bus a pu continuer à "
            "rouler normalement pendant ce silence, ce N'EST PAS une immobilisation confirmée."
        ),
        "chip_dark_gap_between": (
            "{icon} Perte de signal en route : **entre {before} et {after}** (~{min:.0f} min "
            "sans aucun ping)."
        ),
        "chip_dark_gap_after_only": (
            "{icon} Perte de signal en route : **après {before}, plus aucun arrêt suivi jusqu'à "
            "la fin du trajet** (~{min:.0f} min sans aucun ping)."
        ),
        "chip_dark_gap_help": (
            "Trou de signal survenu ENTRE deux arrêts (pas pendant l'attente à un arrêt déjà "
            "repéré) -- invisible au scan arrêt-par-arrêt classique, mais bien réel : le "
            "traceur n'a envoyé AUCUN ping pendant cette durée. Explique généralement à lui "
            "seul le mauvais taux de suivi et la durée gonflée du reste du trajet -- le bus a "
            "très probablement continué de rouler normalement pendant ce silence."
        ),
        "chip_farthest": (
            "{icon} Arrêt suivi mais décalé : **{stop}** (~{dist:.0f} m de la position attendue)."
        ),
        "chip_farthest_help": (
            "Le bus a bien été détecté à cet arrêt, mais sa position GPS était nettement plus "
            "loin que la position officielle de l'arrêt -- déviation d'itinéraire ou dérive GPS."
        ),
        "chip_off_route": "{icon} Arrêts non desservis : {stops}{suffix}.",
        "chip_off_route_help": (
            "Ces arrêts sont dans l'étendue du trajet mais le bus n'y a jamais été détecté à "
            "portée GPS -- trajet partiel, itinéraire différent, ou desserte réellement sautée."
        ),
        "chip_suspect_coord": (
            "{icon} {count} arrêt(s) aux coordonnées douteuses (jamais suivis sur cette ligne — "
            "exclus du diagnostic)."
        ),
        "chip_suspect_coord_help": (
            "Ces arrêts ne sont JAMAIS détectés sur AUCUN trajet de cette ligne -- leurs "
            "coordonnées géographiques sont probablement fausses dans la base de référence, pas "
            "un problème de CE trajet précis. Exclus du diagnostic pour cette raison."
        ),
        "and_others": " (+{n} autres)",

        # Boutons
        "btn_show_map": "Voir la carte du trajet",
        "btn_hide_map": "Masquer la carte",

        # Détour
        "detour_warning": (
            "**Détour non officiel détecté** : le bus a quitté le point de stationnement à "
            "**{left}**, parcouru **~{km:.1f} km** hors de ce point (point le plus éloigné "
            "atteint à **{far}**), puis y est revenu à **{back}** ({total:.0f} min au total) — "
            "probablement une course non planifiée ou un repositionnement, avant "
            "l'immobilisation prolongée signalée ci-dessus.{legs} Les deux trajets GPS réels "
            "sont tracés séparément ci-dessous (couleur orange = aller, violet = retour) pour "
            "suivre le sens du parcours (à ne pas confondre avec l'itinéraire de la ligne, en "
            "gris)."
        ),
        "detour_legs": " Aller : **{left} → {far}** ({out:.0f} min) · Retour : **{far} → {back}** ({back_min:.0f} min).",

        # Carte du trajet
        "map_no_coords": "Coordonnées GPS non disponibles pour les arrêts de cette ligne.",
        "map_hover_time": "Heure de passage : {v}",
        "map_hover_dwell": "Immobilisation réelle : {v}",
        "map_hover_dark": "Signal perdu : {v}",
        "map_hover_dist": "Distance de l'arrêt attendu : {v}",
        "map_hover_tracked": "Suivi GPS : {v}",
        "map_hover_yes": "Oui",
        "map_hover_no": "Non",
        "map_hover_suspect": (
            "<br>Attention : cet arrêt n'est jamais suivi sur cette ligne (coordonnées "
            "probablement erronées)"
        ),
        "detour_leg_out_label": "Détour · aller",
        "detour_leg_back_label": "Détour · retour",
        "legend_normal_stop": "Arrêt normal",
        "legend_long_standstill": "Immobilisation longue",
        "legend_signal_loss": "Perte de signal GPS",
        "legend_unserved": "Arrêt non desservi",
        "legend_suspect_coords": "Coordonnées douteuses",
        "map_legend": (
            "Chaque cercle représente un arrêt, numéroté dans l'ordre de passage (départ → "
            "terminus). La taille reflète la durée d'immobilisation + perte de signal."
        ),
        "map_normal": "Normal",
        "map_long_stop": "Immobilisation longue (≥10 min)",
        "map_signal_loss": "Perte de signal (≥5 min)",
        "map_unserved": "Arrêt non desservi (bus jamais passé)",
        "map_suspect": "Coordonnées d'arrêt douteuses",
        "map_first_last_tracked": (
            "Premier passage suivi : **{t0}** ({stop0}) → dernier passage suivi : **{t1}** ({stop1})"
        ),
        "map_departure": "Départ",
        "map_terminus": "Terminus",
        "map_planned_route": "Itinéraire prévu",

        # Verdict de ligne
        "line_good": "Ligne en bon état",
        "line_watch": "Ligne à surveiller",
        "line_risk": "Ligne à risque élevé",
        "line_verdict_caption": "Ligne {line} · {company} — basé sur {n} trajets analysés.",
        "metric_anomaly_rate": "Taux d'anomalie",
        "metric_flagged_trips": "Trajets signalés",
        "line_not_enough_data": (
            "Ligne {line} : seulement {n} trajet(s) enregistré(s) — pas assez de données pour "
            "juger l'état général de la ligne."
        ),

        # Trajet de référence
        "ref_trip_expander": "Trajet de référence — à quoi ressemble un trajet NORMAL sur la ligne {line} ?",
        "ref_trip_none": "Pas de trajet de référence disponible pour cette ligne.",
        "ref_trip_caption": (
            "Trajet réel, jugé normal par le modèle, choisi parmi les mieux suivis "
            "({match:.0f}% des arrêts) avec une durée proche de la médiane de la ligne pour "
            "cette direction — comparez les anomalies ci-dessous à cette référence."
        ),
        "ref_trip_bus_day": "Bus · Jour",
        "ref_trip_bus": "Bus",
        "ref_trip_duration": "Durée",
        "ref_trip_line_median": "médiane ligne : {med}",
        "ref_trip_stops_tracked": "Arrêts suivis",
        "ref_trip_avg_dwell": "Immobilisation moy.",
        "ref_trip_same_cycle": (
            "🔗 Cycle complet : bus **{bus}**, le **{day}** — l'ALLER et le RETOUR ci-dessous "
            "sont le même bus le même jour, donc la pause au terminus reflète une vraie pause "
            "entre l'arrivée et le redépart."
        ),
        "ref_trip_missing_other_dir": (
            "Direction : **{dir}** (aucun trajet normal exploitable trouvé pour l'autre "
            "direction sur cette ligne)."
        ),
        "ref_typical_idle": (
            "{icon} Stationnement terminus typique pour cette direction : **~{typ:.0f} min** "
            "avant le vrai départ / après la vraie arrivée (médiane sur les trajets normaux). "
            "Au-delà d'environ **{thr:.0f} min** (~2x la normale), un stationnement observé est "
            "probablement un **service non clôturé** (chauffeur n'ayant pas coupé le traceur) "
            "plutôt qu'une pause normale."
        ),

        "btn_analyze_fail": "Impossible de récupérer l'explication.",
        "no_anomaly_found": "{scope} : aucun trajet anormal détecté — tout est dans la normale.",
        "analysis_header": "{scope} — analyse des anomalies",
        "scope_bus_line": "Bus {bus} · Ligne {line}",
        "scope_line": "Ligne {line}",
        "metric_trips_analyzed": "Trajets analysés",
        "metric_trips_analyzed_help": "Nombre total de trajets dans la période sélectionnée pour ce périmètre.",
        "metric_abnormal_trips": "Trajets anormaux",
        "metric_abnormal_trips_help": "Trajets signalés comme anormaux par le modèle de détection (Isolation Forest + LSTM).",
        "metric_normal_duration": "Durée normale (médiane)",
        "metric_normal_duration_help": "Durée médiane d'un trajet non anormal sur cette ligne — sert de référence pour juger si un trajet est trop long ou trop court.",
        "section_flagged_trips": "Trajets signalés",
        "section_trip_detail": "Analyse détaillée d'un trajet",
        "no_trip_matches_filter": "Aucun trajet ne correspond au filtre sélectionné ci-dessus.",
        "select_trip_prompt": "Sélectionnez un trajet ci-dessous pour voir la carte de ses arrêts, le graphique d'immobilisation et le tableau complet arrêt par arrêt.",
        "trip_to_analyze": "Trajet à analyser :",
        "metric_delta_vs_normal": "Écart vs normal",
        "metric_delta_vs_normal_help": "Différence entre ce trajet et la durée normale de la ligne. Un écart positif signifie un trajet plus long que d'habitude.",
        "metric_stops_served": "Arrêts desservis",
        "metric_stops_served_help": "Nombre d'arrêts où le bus a été détecté dans la zone GPS (suivi) sur le total d'arrêts prévus sur la ligne.",
        "metric_anomaly_score": "Score anomalie",
        "metric_anomaly_score_help": "Score calculé par le modèle Isolation Forest. Plus il est élevé, plus le comportement du bus s'écarte de la normale.",
        "section_trip_map": "Carte du trajet",
        "section_dwell_chart": "Immobilisation et perte de signal par arrêt",
        "dwell_chart_caption": (
            "Les barres bleues montrent le temps réel où le bus était à l'arrêt avec GPS actif. "
            "Les barres jaunes montrent les périodes sans ping GPS à cet arrêt (signal perdu). "
            "Les barres rouges correspondent aux arrêts non desservis."
        ),
        "series_real_standstill": "Immobilisation réelle (GPS actif)",
        "series_signal_lost": "Signal perdu (sans ping)",
        "hover_real_standstill": "<b>%{x}</b><br>Immob. réelle : %{y:.1f} min<extra></extra>",
        "hover_signal_lost": "<b>%{x}</b><br>Signal perdu : %{y:.1f} min<extra></extra>",
        "axis_minutes": "Minutes",
        "section_stop_table": "Tableau arrêt par arrêt",
        "stop_table_caption": (
            "**Suivi GPS** : le bus est passé dans la zone de l'arrêt et a été détecté. "
            "**Immob. réelle** : temps d'arrêt avec GPS actif. "
            "**Signal perdu** : durée sans ping GPS à cet arrêt (non comptée comme arrêt). "
            "**Distance arrêt** : écart entre la position GPS du bus et la position théorique de l'arrêt."
        ),
        "col_stop": "Arrêt", "col_gps_tracked": "Suivi GPS", "col_real_standstill": "Immob. réelle",
        "col_signal_lost": "Signal perdu", "col_stop_distance": "Distance arrêt",
        "val_tracked": "Suivi", "val_unserved": "Non desservi",
        "no_sequence_data": "Aucune donnée de séquence disponible pour ce trajet.",

        "route_demo_title": "Démo — Créer une ligne",
        "route_demo_caption": (
            "Dessinez un itinéraire sur la carte (rien n'est enregistré) et comparez le temps "
            "de trajet réel Google à une estimation basée sur la vitesse habituelle de votre flotte."
        ),
        "route_demo_no_key_warning": "Aucune clé Google Maps API configurée pour cette démo.",
        "route_demo_no_key_howto": "Comment obtenir une clé (gratuite)",
        "route_demo_no_key_steps": (
            "1. Créez un projet sur console.cloud.google.com\n"
            "2. Activez **Maps JavaScript API** et **Directions API**\n"
            "3. Ajoutez une carte bancaire (obligatoire même pour le palier gratuit — 200$/mois "
            "de crédit offert, largement suffisant pour une démo)\n"
            "4. Créez une clé API sous *Identifiants*, restreignez-la par référent HTTP "
            "(ex. `localhost:8501/*`) et aux deux API ci-dessus\n"
            "5. Collez la clé ci-dessous, ou ajoutez-la dans `.streamlit/secrets.toml` "
            "(`GOOGLE_MAPS_API_KEY = \"...\"`) pour ne pas avoir à la recoller à chaque fois."
        ),
        "route_demo_paste_key": "Coller la clé API pour cette session",
        "route_demo_stops_header": "Arrêts",
        "route_demo_no_stops_yet": "Cliquez sur la carte pour placer le terminus, puis les arrêts suivants.",
        "route_demo_terminal_label": "Terminus",
        "route_demo_stop_label": "Arrêt",
        "route_demo_undo_btn": "↩️ Annuler le dernier",
        "route_demo_clear_btn": "🗑️ Tout effacer",
        "route_demo_departure_time": "Heure de départ",
        "route_demo_finish_btn": "✅ Terminé — calculer l'horaire",
        "route_demo_cache_hit": "Itinéraire déjà calculé — résultat en cache (0 appel Google).",
        "route_demo_map_help": "Cliquez sur la carte pour ajouter un arrêt, en commençant par le terminus (T).",
        "route_demo_timetable_header": "Horaire estimé",
        "route_demo_timetable_caption": "{dist} km, {n_stops} arrêt(s) au total.",
        "route_demo_col_google": "Heure (Google Maps)",
        "route_demo_col_fleet": "Heure (vitesse flotte)",
        "route_demo_metric_distance": "Distance totale",
        "route_demo_metric_google_eta": "Durée Google Maps",
        "route_demo_metric_fleet_eta": "Durée vitesse flotte",
        "route_demo_new_line_note": (
            "Colonne « vitesse flotte » : basée sur la vitesse commerciale moyenne mesurée sur "
            "l'historique réel de vos bus (~{speed} km/h, {dwell}s d'arrêt moyen par station) — "
            "pas sur l'historique de CETTE ligne précise, puisqu'elle vient d'être créée. Google "
            "Maps donne un temps de conduite réel (voiture) ; la vitesse flotte donne une idée "
            "plus réaliste pour un bus qui s'arrête à chaque station."
        ),
    },
    "en": {
        "page_title_anomaly": "Anomaly detection",
        "page_subtitle_anomaly": "Trips flagged by Isolation Forest + LSTM autoencoder — explained in plain language.",
        "tab_live": "Flagged trips",
        "tab_explain": "Investigate a bus",
        "tab_patterns": "Trends",
        "tab_tickets": "Ticket anomalies",
        "lang_label": "Language",

        "filter_operator": "Operator",
        "filter_line": "Line",
        "filter_direction": "Direction",
        "filter_both_directions": "Both",
        "filter_all_lines": "All lines",
        "filter_bus": "Bus",
        "filter_all_buses": "All buses",
        "filter_day": "Day",
        "filter_all_days": "All days",
        "filter_anomaly_type": "Filter by anomaly type",
        "sort_by": "Sort by",
        "sort_date_desc": "📅 Date (most recent first)",
        "sort_date_asc": "📅 Date (oldest first)",
        "sort_severity_desc": "🚨 Severity (highest first)",
        "sort_severity_asc": "🚨 Severity (lowest first)",
        "sort_duration_desc": "⏱️ Duration (longest first)",
        "btn_analyze": "Analyze",
        "show_data_bugs": "Also show data bugs / tracking fragments",

        "model_warning_neither": (
            "This operator doesn't yet have enough recorded trips for a dedicated anomaly "
            "model (neither Isolation Forest nor the LSTM autoencoder). Detection below is "
            "therefore comparing these trips against the whole network (all operators and "
            "lines pooled), **not specifically this line or this operator** -- the result "
            "isn't guaranteed accurate and should be read with caution. Accuracy will improve "
            "automatically as this operator accumulates more data in future updates."
        ),
        "model_warning_lstm_only": (
            "This operator has enough trips for its own Isolation Forest model, but not yet "
            "enough (under 200 trips) for a dedicated LSTM autoencoder -- that one falls back "
            "to a **global** model trained across all operators. And even the \"dedicated\" "
            "Isolation Forest is trained on ALL of this operator's lines pooled together, not "
            "this specific line. The result isn't guaranteed accurate and should be read with "
            "caution -- accuracy will improve automatically with more data in future updates."
        ),

        "tk_view_label": "View",
        "tk_view_admin": "Admin (all)",
        "tk_view_client": "Client (reliable only)",
        "tk_machine_issue": (
            "Likely ticket-machine fault: the bus did run that day ({trips} confirmed GPS "
            "trip(s)) but almost no ticket was recorded. This is almost certainly not a "
            "revenue/fraud anomaly -- hidden in the client view."
        ),
        "tk_no_service": (
            "No GPS trips that day for this bus -- a no-service day (holiday, strike, bus out "
            "of service), not a revenue anomaly. Hidden in the client view."
        ),
        "tk_good_anomaly": (
            "Good anomaly: this day's revenue is above normal (for this bus, or the line if "
            "not enough history for this bus) -- the model flags it as statistically unusual, "
            "but more money coming in than usual isn't a problem to act on."
        ),
        "tk_bad_anomaly": (
            "Anomaly worth checking: this day's revenue is below normal (for this bus, or the "
            "line if not enough history for this bus)."
        ),
        "tk_machine_detection_explain": (
            "How we tell a machine fault apart from a genuine anomaly: every flagged "
            "ticket-day is cross-checked against THIS bus's real GPS trips that same day. "
            "If GPS confirms the bus ran but almost no tickets (2 or fewer) were recorded, "
            "it's almost certainly that the machine didn't work -- not a revenue anomaly. If "
            "GPS shows NO trips at all that day, it's a no-service day (holiday, strike, bus "
            "out of service), not an anomaly either. Either way: automatically excluded from "
            "the client view, shown with this explanation in the admin view."
        ),
        "tk_filter_label": "Priority",
        "tk_filter_bad": "📉 Worth checking (low revenue)",
        "tk_filter_good": "📈 Good (high revenue)",
        "tk_filter_all": "All",

        "section_flagged_today": "Trips flagged today",
        "section_recent_history": "Recent history",
        "metric_operating_day": "Operating day",
        "live_data_badge": "LIVE data (GPS web service, today)",
        "historical_data_badge": "Historical data (web service unavailable/not ready -- last known day in the dataset)",
        "metric_trips_today": "Trips today",
        "metric_flagged": "Flagged",
        "no_anomaly_today": "No anomalies on the latest operating day for this scope.",
        "try_specific_line": "Try a specific line, or check the history below.",
        "no_history": "No historical anomalies for this scope.",

        "metric_trip_duration": "Trip duration",
        "metric_departure_arrival": "Departure → Arrival",
        "metric_verifiable_activity": "Verifiable GPS activity",
        "metric_severity": "Severity",
        "sev_high": "High",
        "sev_medium": "Medium",
        "sev_low": "Low",
        "main_cause": "Main cause: **{label}**",
        "other_uncategorized": "Other / uncategorized",
        "flagged_no_reason": "Flagged by the model score (no single dominant cause).",

        "topfeat_max_dwell_s": "Abnormal standstill",
        "topfeat_total_elapsed": "Abnormally long trip",
        "topfeat_mean_dwell_s": "High average stop duration",
        "topfeat_dist_m_max": "Route deviation",
        "topfeat_match_rate": "Poor GPS tracking / off-route",
        "topfeat_n_stops": "Abnormally few stops served",
        "topfeat_max_dark_s": "GPS signal loss",
        "topfeat_terminus_idle_min": "Terminus dwell (service not closed out)",
        "topfeat_elapsed_vs_bus_z": "Unusual duration for this bus",
        "topfeat_elapsed_vs_line_z": "Unusual duration for this line",

        "formula_caption": (
            "**Formula**: Trip duration = arrival time ({arr}) − departure time ({dep}), "
            "**after** removing {idle:.0f} min of stationary time at the edges of the trip "
            "(before the real departure / after the real arrival). It does **NOT exclude** "
            "stops or standstills that happened DURING the trip (those are part of the actual "
            "service and stay counted in the duration)."
        ),
        "formula_help": (
            "Trip duration = last ping − first ping, AFTER removing stationary time at the trip "
            "edges (bus parked before really departing / after really arriving). A stop or "
            "standstill occurring DURING the trip, though, stays counted in this duration -- "
            "that's a different thing."
        ),
        "verifiable_activity_caption": (
            "**Verifiable GPS activity ≈ {est}**  ({dur} − {dwell:.0f} min stationary − "
            "{dark:.0f} min without signal), with {match:.0f}% of stops tracked. Lower-bound "
            "estimate: driving time during signal gaps and stationary time away from known "
            "stops can't be counted."
        ),

        "q_data_bug": (
            "**Data bug**: duration > 24h is physically impossible — corrupted timestamps in "
            "the GPS source (not a real trip). Will be permanently eliminated on the next "
            "rebuild."
        ),
        "q_fragment": (
            "**Tracking fragment**: duration far below the line's normal with almost no stops "
            "tracked — partial trip or patchy GPS coverage, not comparable to a full trip."
        ),
        "q_dark_inflated": (
            "**Why this duration?** Most of this \"duration\" is a GPS signal gap: the tracker "
            "went silent mid-route and came back on much later. The bus most likely finished "
            "its service normally during that silence — the displayed duration reflects the "
            "tracker's behavior, not real driving time. This is not an error in the anomaly "
            "model: it correctly flags a failed GPS tracking, which should be treated as such."
        ),
        "q_implausible": (
            "**Why this duration?** The GPS tracker kept transmitting after the service ended "
            "(driver didn't close out the service / bus parked at the terminus or depot) — this "
            "parked time got merged into the trip. Real driving time is significantly shorter. "
            "This isn't a flaw in the model or the segmentation: it's an operational practice "
            "(service not closed out), automatically corrected on the next data rebuild."
        ),
        "q_partial_coverage": (
            "**Why this duration?** This trip only covered **{ns} stop(s)** against "
            "**{mns:.0f} typically** on this line/direction — the bus genuinely covered a "
            "shorter distance, so a shorter duration is normal for THIS specific trip. This "
            "isn't a speed anomaly: comparing it to the median of full trips would be unfair, "
            "like comparing a one-way trip to a round trip."
        ),

        "exp_max_dwell_s": (
            "The longest stationary stop detected on this trip is well above normal. Measured "
            "from GPS pings actually received (bus truly motionless), not a signal gap."
        ),
        "exp_total_elapsed": (
            "Total trip duration (arrival − departure), after removing stationary time at the "
            "trip edges (see Formula) -- already cleaned of service-not-closed-out at both "
            "ends, but NOT of stops that happened along the way."
        ),
        "exp_mean_dwell_s": (
            "Average stop dwell time on this trip, higher than the line's normal -- suggests "
            "widespread slowdown (traffic, cascading delays) rather than a single-stop incident."
        ),
        "exp_dist_m_max": (
            "Largest gap observed between the bus's GPS position and a stop's official position "
            "that was still counted as served -- route deviation or GPS drift."
        ),
        "exp_match_rate": (
            "Share of the line's stops actually detected by GPS on this trip. A low rate can "
            "come from a partial trip, poor GPS tracking, or stops with unreliable coordinates "
            "(see chips below)."
        ),
        "exp_n_stops": (
            "Number of stops covered by this trip, well below the line/direction's normal -- "
            "points to a partial trip rather than a speed issue."
        ),
        "exp_max_dark_s": (
            "The longest GPS signal gap detected (no pings received) on this trip. The bus may "
            "have kept driving normally during that silence -- this time is NOT counted as a "
            "standstill, only as uncertainty."
        ),
        "exp_terminus_idle_min": (
            "Time the GPS tracker kept pinging while the bus was parked at the terminus -- "
            "before the real departure or after the real arrival. This time is ALREADY REMOVED "
            "from the displayed trip duration (see Formula); this figure shows it separately."
        ),
        "exp_elapsed_vs_bus_z": (
            "Standard-deviation score (z-score) of THIS trip's duration against THIS specific "
            "bus's history on this line. The compared duration already excludes stationary time "
            "at the trip edges (see Formula) -- this isn't an un-removed parking artifact."
        ),
        "exp_elapsed_vs_line_z": (
            "Standard-deviation score (z-score) of THIS trip's duration against the median of "
            "ALL buses on this line/direction. The compared duration already excludes "
            "stationary time at the trip edges (see Formula) -- this isn't an un-removed "
            "parking artifact."
        ),
        "exp_default": "Signal used by the detection model to flag this trip as abnormal.",

        "chip_terminus_idle": (
            "{icon} Terminus dwell: **{min:.0f} min** engine/tracker active before departure or "
            "after arrival (service likely not closed out — not counted in the trip duration "
            "above)."
        ),
        "chip_terminus_idle_help": (
            "This time is REMOVED from the \"Trip duration\" shown at the top of the card -- "
            "the bus was still pinging but hadn't really started/ended its service. A genuine "
            "operational signal (driver likely didn't turn off the tracker), not a measurement "
            "error."
        ),
        "chip_origin_idle": (
            "{icon} Terminus dwell at **{stop}** before departure: **{min:.0f} min** -- "
            "stationary from {from_t} to {to_t}, actual departure at {to_t} (service likely "
            "not closed out — not counted in the trip duration above)."
        ),
        "chip_origin_idle_help": (
            "This time is REMOVED from the \"Trip duration\" shown at the top of the card -- "
            "the bus was already pinging at the DEPARTURE terminus but hadn't started its "
            "service yet. A genuine operational signal (driver likely didn't turn off the "
            "tracker), not a measurement error."
        ),
        "chip_end_idle": (
            "{icon} Terminus dwell at **{stop}** after arrival: **{min:.0f} min** -- "
            "stationary from {from_t} to {to_t} (service likely not closed out — not counted "
            "in the trip duration above)."
        ),
        "chip_end_idle_help": (
            "This time is REMOVED from the \"Trip duration\" shown at the top of the card -- "
            "the bus was still pinging at the ARRIVAL terminus but hadn't really ended its "
            "service. A genuine operational signal (driver likely didn't turn off the "
            "tracker), not a measurement error."
        ),
        "chip_real_stop": "{icon} Genuine standstill: **{stop}** ({min:.0f} min without GPS movement).",
        "chip_real_stop_help": (
            "Standstill detected DURING the trip (GPS pings present, bus genuinely motionless) "
            "-- stays counted in the \"Trip duration\" above, unlike terminus dwell time which "
            "is removed from it."
        ),
        "chip_detour_hint": (
            "&nbsp;&nbsp;↳ *If this delay follows the terminus dwell above, it may not be the "
            "same stop: the bus could have left and come back before settling down again "
            "(unofficial detour / unplanned errand). \"View trip map\" checks the raw GPS pings "
            "and shows the actual route if that's the case.*"
        ),
        "chip_same_terminus_hint": (
            "&nbsp;&nbsp;↳ *This is the SAME stop as the terminus dwell above (**{stop}**) -- "
            "most likely ONE continuous standstill split in two by a brief isolated GPS blip, "
            "not two separate events. Likely actual duration: ~{total:.0f} min.*"
        ),
        "chip_signal_loss": "{icon} Signal loss: **{stop}** (~{min:.0f} min without a ping).",
        "chip_signal_loss_help": (
            "No GPS ping received for this long at this stop -- the bus may have kept driving "
            "normally during that silence, this is NOT a confirmed standstill."
        ),
        "chip_dark_gap_between": (
            "{icon} Signal lost en route: **between {before} and {after}** (~{min:.0f} min with "
            "no ping at all)."
        ),
        "chip_dark_gap_after_only": (
            "{icon} Signal lost en route: **after {before}, no further stop was tracked for the "
            "rest of the trip** (~{min:.0f} min with no ping at all)."
        ),
        "chip_dark_gap_help": (
            "A signal gap that happened BETWEEN two stops (not while waiting at a stop already "
            "matched) -- invisible to the regular stop-by-stop scan, but very real: the tracker "
            "sent NO ping at all for this long. This alone usually explains the rest of the "
            "trip's poor tracking rate and inflated duration -- the bus most likely kept driving "
            "normally during this silence."
        ),
        "chip_farthest": "{icon} Tracked but off position: **{stop}** (~{dist:.0f} m from the expected spot).",
        "chip_farthest_help": (
            "The bus was detected at this stop, but its GPS position was significantly farther "
            "than the stop's official location -- route deviation or GPS drift."
        ),
        "chip_off_route": "{icon} Unserved stops: {stops}{suffix}.",
        "chip_off_route_help": (
            "These stops are within the trip's span but the bus was never detected in GPS range "
            "of them -- partial trip, a different route, or a genuinely skipped stop."
        ),
        "chip_suspect_coord": (
            "{icon} {count} stop(s) with questionable coordinates (never tracked on this line — "
            "excluded from the diagnosis)."
        ),
        "chip_suspect_coord_help": (
            "These stops are NEVER detected on ANY trip of this line -- their geographic "
            "coordinates are likely wrong in the reference data, not an issue with THIS "
            "specific trip. Excluded from the diagnosis for that reason."
        ),
        "and_others": " (+{n} more)",

        "btn_show_map": "View trip map",
        "btn_hide_map": "Hide map",

        "detour_warning": (
            "**Unofficial detour detected**: the bus left the parked spot at **{left}**, "
            "traveled **~{km:.1f} km** away from it (farthest point reached at **{far}**), then "
            "came back at **{back}** ({total:.0f} min total) — likely an unplanned errand or "
            "repositioning, before the extended standstill flagged above.{legs} Both real GPS "
            "paths are traced separately below (orange = outbound, purple = return) to follow "
            "the direction of travel (not to be confused with the line's route, in gray)."
        ),
        "detour_legs": " Outbound: **{left} → {far}** ({out:.0f} min) · Return: **{far} → {back}** ({back_min:.0f} min).",

        "map_no_coords": "GPS coordinates not available for this line's stops.",
        "map_hover_time": "Time: {v}",
        "map_hover_dwell": "Genuine standstill: {v}",
        "map_hover_dark": "Signal lost: {v}",
        "map_hover_dist": "Distance from expected stop: {v}",
        "map_hover_tracked": "GPS tracked: {v}",
        "map_hover_yes": "Yes",
        "map_hover_no": "No",
        "map_hover_suspect": (
            "<br>Warning: this stop is never tracked on this line (coordinates are likely wrong)"
        ),
        "detour_leg_out_label": "Detour · outbound",
        "detour_leg_back_label": "Detour · return",
        "legend_normal_stop": "Normal stop",
        "legend_long_standstill": "Long standstill",
        "legend_signal_loss": "GPS signal loss",
        "legend_unserved": "Unserved stop",
        "legend_suspect_coords": "Questionable coordinates",
        "map_legend": (
            "Each circle represents a stop, numbered in visiting order (departure → terminus). "
            "Size reflects standstill duration + signal loss."
        ),
        "map_normal": "Normal",
        "map_long_stop": "Long standstill (≥10 min)",
        "map_signal_loss": "Signal loss (≥5 min)",
        "map_unserved": "Unserved stop (bus never passed)",
        "map_suspect": "Questionable stop coordinates",
        "map_first_last_tracked": (
            "First tracked passage: **{t0}** ({stop0}) → last tracked passage: **{t1}** ({stop1})"
        ),
        "map_departure": "Departure",
        "map_terminus": "Terminus",
        "map_planned_route": "Planned route",

        "line_good": "Line in good shape",
        "line_watch": "Line to watch",
        "line_risk": "High-risk line",
        "line_verdict_caption": "Line {line} · {company} — based on {n} analyzed trips.",
        "metric_anomaly_rate": "Anomaly rate",
        "metric_flagged_trips": "Flagged trips",
        "line_not_enough_data": (
            "Line {line}: only {n} recorded trip(s) — not enough data to judge the line's "
            "overall health."
        ),

        "ref_trip_expander": "Reference trip — what does a NORMAL trip look like on line {line}?",
        "ref_trip_none": "No reference trip available for this line.",
        "ref_trip_caption": (
            "A real trip, judged normal by the model, chosen among the best-tracked "
            "({match:.0f}% of stops) with a duration close to the line's median for this "
            "direction — compare the anomalies below to this reference."
        ),
        "ref_trip_bus_day": "Bus · Day",
        "ref_trip_bus": "Bus",
        "ref_trip_duration": "Duration",
        "ref_trip_line_median": "line median: {med}",
        "ref_trip_stops_tracked": "Stops tracked",
        "ref_trip_avg_dwell": "Avg. standstill",
        "ref_trip_same_cycle": (
            "🔗 Full cycle: bus **{bus}**, on **{day}** — the OUTBOUND and RETURN below are "
            "the same bus on the same day, so the terminus standstill reflects a real break "
            "between arrival and departure."
        ),
        "ref_trip_missing_other_dir": (
            "Direction: **{dir}** (no usable normal trip found for the other direction on this "
            "line)."
        ),
        "ref_typical_idle": (
            "{icon} Typical terminus dwell for this direction: **~{typ:.0f} min** before the "
            "real departure / after the real arrival (median over normal trips). Beyond roughly "
            "**{thr:.0f} min** (~2x normal), an observed standstill is likely a **service not "
            "closed out** (driver probably didn't turn off the tracker) rather than a normal "
            "pause."
        ),

        "btn_analyze_fail": "Could not retrieve the explanation.",
        "no_anomaly_found": "{scope}: no abnormal trip detected — everything is within normal range.",
        "analysis_header": "{scope} — anomaly analysis",
        "scope_bus_line": "Bus {bus} · Line {line}",
        "scope_line": "Line {line}",
        "metric_trips_analyzed": "Trips analyzed",
        "metric_trips_analyzed_help": "Total number of trips in the selected period for this scope.",
        "metric_abnormal_trips": "Abnormal trips",
        "metric_abnormal_trips_help": "Trips flagged as abnormal by the detection model (Isolation Forest + LSTM).",
        "metric_normal_duration": "Normal duration (median)",
        "metric_normal_duration_help": "Median duration of a non-abnormal trip on this line — used as a reference to judge whether a trip is too long or too short.",
        "section_flagged_trips": "Flagged trips",
        "section_trip_detail": "Detailed analysis of a trip",
        "no_trip_matches_filter": "No trip matches the filter selected above.",
        "select_trip_prompt": "Select a trip below to see its stop map, the standstill chart, and the full stop-by-stop table.",
        "trip_to_analyze": "Trip to analyze:",
        "metric_delta_vs_normal": "Deviation from normal",
        "metric_delta_vs_normal_help": "Difference between this trip and the line's normal duration. A positive value means a longer-than-usual trip.",
        "metric_stops_served": "Stops served",
        "metric_stops_served_help": "Number of stops where the bus was detected in GPS range (tracked), out of the line's total planned stops.",
        "metric_anomaly_score": "Anomaly score",
        "metric_anomaly_score_help": "Score computed by the Isolation Forest model. The higher it is, the more the bus's behavior deviates from normal.",
        "section_trip_map": "Trip map",
        "section_dwell_chart": "Standstill and signal loss by stop",
        "dwell_chart_caption": (
            "Blue bars show the real time the bus was stopped with GPS active. Yellow bars show "
            "periods without a GPS ping at that stop (signal lost). Red bars correspond to "
            "unserved stops."
        ),
        "series_real_standstill": "Genuine standstill (GPS active)",
        "series_signal_lost": "Signal lost (no ping)",
        "hover_real_standstill": "<b>%{x}</b><br>Standstill: %{y:.1f} min<extra></extra>",
        "hover_signal_lost": "<b>%{x}</b><br>Signal lost: %{y:.1f} min<extra></extra>",
        "axis_minutes": "Minutes",
        "section_stop_table": "Stop-by-stop table",
        "stop_table_caption": (
            "**GPS tracked**: the bus passed through the stop's zone and was detected. **Genuine "
            "standstill**: stopped time with GPS active. **Signal lost**: time without a GPS ping "
            "at this stop (not counted as a standstill). **Stop distance**: gap between the bus's "
            "GPS position and the stop's official position."
        ),
        "col_stop": "Stop", "col_gps_tracked": "GPS tracked", "col_real_standstill": "Genuine standstill",
        "col_signal_lost": "Signal lost", "col_stop_distance": "Stop distance",
        "val_tracked": "Tracked", "val_unserved": "Unserved",
        "no_sequence_data": "No sequence data available for this trip.",
    },
    "ar": {
        "page_title_anomaly": "كشف الانحراف",
        "page_subtitle_anomaly": "رحلات أشار إليها ⁦Isolation Forest⁩ + مُرمِّز ⁦LSTM⁩ التلقائي — مُفسَّرة بلغة واضحة.",
        "tab_live": "الرحلات المُبلَّغ عنها",
        "tab_explain": "تحليل حافلة",
        "tab_patterns": "الاتجاهات",
        "tab_tickets": "انحراف التذاكر",
        "lang_label": "اللغة",

        "filter_operator": "المشغّل",
        "filter_line": "الخط",
        "filter_direction": "الاتجاه",
        "filter_both_directions": "كلا الاتجاهين",
        "filter_all_lines": "جميع الخطوط",
        "filter_bus": "الحافلة",
        "filter_all_buses": "جميع الحافلات",
        "filter_day": "اليوم",
        "filter_all_days": "جميع الأيام",
        "filter_anomaly_type": "تصفية حسب نوع الانحراف",
        "sort_by": "ترتيب حسب",
        "sort_date_desc": "📅 التاريخ (الأحدث أولاً)",
        "sort_date_asc": "📅 التاريخ (الأقدم أولاً)",
        "sort_severity_desc": "🚨 الخطورة (الأعلى أولاً)",
        "sort_severity_asc": "🚨 الخطورة (الأدنى أولاً)",
        "sort_duration_desc": "⏱️ المدة (الأطول أولاً)",
        "btn_analyze": "تحليل",
        "show_data_bugs": "إظهار أخطاء البيانات / أجزاء التتبع الناقصة أيضًا",

        "model_warning_neither": (
            "لا يزال لدى هذا المشغّل عدد غير كافٍ من الرحلات المسجّلة لنموذج انحراف مخصص "
            "(لا ⁦Isolation Forest⁩ ولا مُرمِّز ⁦LSTM⁩ التلقائي). لذلك يقارن الكشف أدناه هذه "
            "الرحلات بالشبكة بأكملها (كل الشركات والخطوط مجتمعة)، **وليس تحديدًا بهذا الخط "
            "أو هذا المشغّل** -- النتيجة غير مضمونة الدقة ويجب قراءتها بحذر. ستتحسن الدقة "
            "تلقائيًا مع تراكم المزيد من بيانات هذا المشغّل في التحديثات القادمة."
        ),
        "model_warning_lstm_only": (
            "لدى هذا المشغّل عدد كافٍ من الرحلات لنموذج ⁦Isolation Forest⁩ الخاص به، لكن ليس "
            "بعد كافيًا (أقل من 200 رحلة) لمُرمِّز ⁦LSTM⁩ تلقائي مخصص -- فيعتمد على نموذج "
            "**عام** مدرَّب على كل المشغّلين. وحتى ⁦Isolation Forest⁩ \"المخصص\" مدرَّب على "
            "كل خطوط هذا المشغّل مجتمعة، وليس على هذا الخط تحديدًا. النتيجة غير مضمونة "
            "الدقة ويجب قراءتها بحذر -- ستتحسن الدقة تلقائيًا مع مزيد من البيانات في "
            "التحديثات القادمة."
        ),

        "tk_view_label": "العرض",
        "tk_view_admin": "المسؤول (الكل)",
        "tk_view_client": "العميل (الموثوق فقط)",
        "tk_machine_issue": (
            "عطل محتمل في آلة التذاكر: الحافلة سارت فعلاً في ذلك اليوم (⁦{trips}⁩ رحلة ⁦GPS⁩ "
            "مؤكدة) لكن لم يُسجَّل أي تذكرة تقريبًا. هذا على الأرجح ليس انحرافًا في الإيرادات/"
            "احتيالًا -- مخفي في عرض العميل."
        ),
        "tk_no_service": (
            "لا توجد رحلات ⁦GPS⁩ في ذلك اليوم لهذه الحافلة -- يوم بلا خدمة (عطلة، إضراب، "
            "الحافلة خارج الخدمة)، وليس انحرافًا في الإيرادات. مخفي في عرض العميل."
        ),
        "tk_good_anomaly": (
            "انحراف جيد: إيرادات هذا اليوم أعلى من المعتاد (لهذه الحافلة، أو للخط إذا لم يكن "
            "هناك سجل كافٍ لهذه الحافلة) -- يُشير إليه النموذج لأنه غير معتاد إحصائيًا، لكن "
            "دخول مال أكثر من المعتاد ليس مشكلة يجب معالجتها."
        ),
        "tk_bad_anomaly": (
            "انحراف يستحق المتابعة: إيرادات هذا اليوم أقل من المعتاد (لهذه الحافلة، أو للخط "
            "إذا لم يكن هناك سجل كافٍ لهذه الحافلة)."
        ),
        "tk_machine_detection_explain": (
            "كيف نميّز عطل الآلة عن انحراف حقيقي: يُقارَن كل يوم تذاكر مُبلَّغ عنه برحلات ⁦GPS⁩ "
            "الفعلية لهذه الحافلة في نفس اليوم. إذا أكّد ⁦GPS⁩ أن الحافلة سارت لكن لم يُسجَّل "
            "أي تذكرة تقريبًا (تذكرتان أو أقل)، فمن شبه المؤكد أن الآلة لم تعمل -- وليس "
            "انحرافًا في الإيرادات. وإذا لم يُظهر ⁦GPS⁩ أي رحلة إطلاقًا ذلك اليوم، فهو يوم بلا "
            "خدمة (عطلة، إضراب، حافلة خارج الخدمة)، وليس انحرافًا أيضًا. في الحالتين: يُستبعد "
            "تلقائيًا من عرض العميل، ويظهر مع هذا التوضيح في عرض المسؤول."
        ),
        "tk_filter_label": "الأولوية",
        "tk_filter_bad": "📉 يستحق المتابعة (إيرادات منخفضة)",
        "tk_filter_good": "📈 جيدة (إيرادات مرتفعة)",
        "tk_filter_all": "الكل",

        "section_flagged_today": "الرحلات المُبلَّغ عنها اليوم",
        "section_recent_history": "السجل الأخير",
        "metric_operating_day": "يوم التشغيل",
        "live_data_badge": "بيانات مباشرة (خدمة ⁦GPS⁩ الإلكترونية، اليوم نفسه)",
        "historical_data_badge": "بيانات تاريخية (الخدمة الإلكترونية غير متاحة/غير جاهزة -- آخر يوم معروف في البيانات)",
        "metric_trips_today": "رحلات اليوم",
        "metric_flagged": "المُبلَّغ عنها",
        "no_anomaly_today": "لا توجد حالات انحراف في آخر يوم تشغيل لهذا النطاق.",
        "try_specific_line": "جرّب اختيار خط محدد، أو راجع السجل أدناه.",
        "no_history": "لا توجد حالات انحراف سابقة لهذا النطاق.",

        "metric_trip_duration": "مدة الرحلة",
        "metric_departure_arrival": "المغادرة ← الوصول",
        "metric_verifiable_activity": "نشاط ⁦GPS⁩ القابل للتحقق",
        "metric_severity": "الخطورة",
        "sev_high": "مرتفعة",
        "sev_medium": "متوسطة",
        "sev_low": "منخفضة",
        "main_cause": "السبب الرئيسي: **⁦{label}⁩**",
        "other_uncategorized": "أخرى / غير مصنّفة",
        "flagged_no_reason": "أُبلغ عنها بواسطة درجة النموذج (لا يوجد سبب واحد مهيمن).",

        "topfeat_max_dwell_s": "توقف غير طبيعي",
        "topfeat_total_elapsed": "رحلة أطول من المعتاد",
        "topfeat_mean_dwell_s": "متوسط مدة التوقف مرتفع",
        "topfeat_dist_m_max": "انحراف عن المسار",
        "topfeat_match_rate": "تتبع ⁦GPS⁩ ضعيف / خارج المسار",
        "topfeat_n_stops": "عدد محطات مخدومة أقل من المعتاد",
        "topfeat_max_dark_s": "انقطاع إشارة ⁦GPS⁩",
        "topfeat_terminus_idle_min": "توقف بالمحطة النهائية (خدمة لم تُغلق)",
        "topfeat_elapsed_vs_bus_z": "مدة غير معتادة لهذه الحافلة",
        "topfeat_elapsed_vs_line_z": "مدة غير معتادة لهذا الخط",

        "formula_caption": (
            "**المعادلة**: مدة الرحلة = وقت الوصول (⁦{arr}⁩) − وقت المغادرة (⁦{dep}⁩)، "
            "**بعد** طرح ⁦{idle:.0f}⁩ دقيقة من التوقف عند طرفي الرحلة (قبل المغادرة الفعلية "
            "/ بعد الوصول الفعلي). وهي **لا تستثني** التوقفات التي حدثت أثناء الرحلة نفسها "
            "(فهذه جزء من الخدمة الفعلية وتبقى محسوبة ضمن المدة)."
        ),
        "formula_help": (
            "مدة الرحلة = آخر إشارة − أول إشارة، بعد طرح فترة التوقف عند طرفي الرحلة "
            "(حافلة متوقفة قبل المغادرة الفعلية / بعد الوصول الفعلي). أما التوقف الذي يحدث "
            "أثناء الرحلة فيبقى محسوبًا ضمن هذه المدة -- الأمران مختلفان."
        ),
        "verifiable_activity_caption": (
            "**نشاط ⁦GPS⁩ القابل للتحقق ≈ ⁦{est}⁩** (⁦{dur}⁩ − ⁦{dwell:.0f}⁩ دقيقة توقف − "
            "⁦{dark:.0f}⁩ دقيقة بلا إشارة)، مع تتبع ⁦{match:.0f}⁩% من المحطات. تقدير أدنى: "
            "وقت القيادة أثناء انقطاع الإشارة والتوقف خارج المحطات المعروفة لا يمكن احتسابه."
        ),

        "q_data_bug": (
            "**خطأ في البيانات**: مدة تتجاوز 24 ساعة أمر مستحيل فعليًا — طوابع زمنية تالفة "
            "في مصدر ⁦GPS⁩ (ليست رحلة حقيقية). سيُحذف نهائيًا عند إعادة البناء القادمة."
        ),
        "q_fragment": (
            "**جزء تتبع ناقص**: مدة أقل بكثير من معدل الخط مع تتبع شبه معدوم للمحطات — "
            "رحلة جزئية أو تغطية ⁦GPS⁩ ناقصة، لا يمكن مقارنتها برحلة كاملة."
        ),
        "q_dark_inflated": (
            "**لماذا هذه المدة؟** معظم هذه «المدة» هو انقطاع في إشارة ⁦GPS⁩: توقف الجهاز عن "
            "الإرسال في منتصف الطريق ثم عاد للعمل لاحقًا بكثير. على الأرجح أنهت الحافلة "
            "خدمتها بشكل طبيعي أثناء هذا الصمت — المدة المعروضة تعكس سلوك الجهاز، لا وقت "
            "القيادة الفعلي. هذا ليس خطأً من نموذج كشف الانحراف: إنه يشير بشكل صحيح إلى تتبع "
            "⁦GPS⁩ معطل، وينبغي التعامل معه على هذا الأساس."
        ),
        "q_implausible": (
            "**لماذا هذه المدة؟** استمر جهاز ⁦GPS⁩ بالإرسال بعد انتهاء الخدمة (لم يُغلق "
            "السائق الخدمة / الحافلة متوقفة عند المحطة النهائية أو المستودع) — فاندمج وقت "
            "التوقف هذا ضمن الرحلة. وقت القيادة الفعلي أقصر بكثير. هذا ليس خللاً في النموذج "
            "أو في تجزئة الرحلات: إنها ممارسة تشغيلية (خدمة لم تُغلق)، تُصحَّح تلقائيًا عند "
            "إعادة بناء البيانات القادمة."
        ),
        "q_partial_coverage": (
            "**لماذا هذه المدة؟** غطت هذه الرحلة **⁦{ns}⁩ محطة فقط** مقابل **⁦{mns:.0f}⁩ عادةً** "
            "على هذا الخط/الاتجاه — فعليًا قطعت الحافلة مسافة أقصر، لذا فإن مدة أقصر أمر "
            "طبيعي لهذه الرحلة تحديدًا. هذا ليس انحرافًا في السرعة: مقارنتها بمتوسط الرحلات "
            "الكاملة ستكون غير عادلة، كمقارنة رحلة ذهاب فقط برحلة ذهاب وإياب."
        ),

        "exp_max_dwell_s": (
            "أطول توقف مسجَّل في هذه الرحلة يتجاوز المعدل الطبيعي بوضوح. مُقاس من إشارات "
            "⁦GPS⁩ الفعلية (حافلة متوقفة فعليًا)، وليس من انقطاع إشارة."
        ),
        "exp_total_elapsed": (
            "المدة الإجمالية للرحلة (الوصول − المغادرة)، بعد طرح التوقف عند طرفي الرحلة "
            "(انظر المعادلة) -- منقّاة بالفعل من حالات عدم إغلاق الخدمة عند الطرفين، لكن "
            "ليس من التوقفات التي حدثت أثناء الطريق."
        ),
        "exp_mean_dwell_s": (
            "متوسط مدة التوقف عند كل محطة في هذه الرحلة أعلى من معدل الخط -- يشير إلى تباطؤ "
            "عام (ازدحام، تأخيرات متتالية) أكثر منه حادثة في محطة واحدة."
        ),
        "exp_dist_m_max": (
            "أكبر فارق مُسجَّل بين موقع ⁦GPS⁩ للحافلة والموقع الرسمي لمحطة اعتُبرت مع ذلك "
            "مخدومة -- انحراف عن المسار أو انجراف في إشارة ⁦GPS⁩."
        ),
        "exp_match_rate": (
            "نسبة محطات الخط التي رصدها ⁦GPS⁩ فعليًا في هذه الرحلة. قد تنتج النسبة المنخفضة عن "
            "رحلة جزئية، أو تتبع ⁦GPS⁩ ضعيف، أو محطات ذات إحداثيات غير موثوقة (انظر الشارات "
            "أدناه)."
        ),
        "exp_n_stops": (
            "عدد المحطات المخدومة في هذه الرحلة أقل بكثير من معدل الخط/الاتجاه -- يشير إلى "
            "رحلة جزئية وليس مشكلة سرعة."
        ),
        "exp_max_dark_s": (
            "أطول انقطاع لإشارة ⁦GPS⁩ (بدون أي إشارة مستلمة) في هذه الرحلة. قد تكون الحافلة "
            "واصلت السير بشكل طبيعي أثناء هذا الصمت -- هذا الوقت غير محسوب كتوقف، بل كحالة "
            "عدم يقين فقط."
        ),
        "exp_terminus_idle_min": (
            "الوقت الذي استمر فيه جهاز ⁦GPS⁩ بالإرسال بينما كانت الحافلة متوقفة عند المحطة "
            "النهائية -- قبل المغادرة الفعلية أو بعد الوصول الفعلي. هذا الوقت **محذوف بالفعل** "
            "من مدة الرحلة المعروضة (انظر المعادلة)؛ هذا الرقم يعرضه بشكل منفصل."
        ),
        "exp_elapsed_vs_bus_z": (
            "الانحراف المعياري (⁦z-score⁩) لمدة هذه الرحلة مقارنةً بتاريخ هذه الحافلة تحديدًا "
            "على هذا الخط. المدة المُقارنة تستثني بالفعل التوقف عند طرفي الرحلة (انظر "
            "المعادلة) -- ليست أثرًا ناتجًا عن توقف لم يُحذف."
        ),
        "exp_elapsed_vs_line_z": (
            "الانحراف المعياري (⁦z-score⁩) لمدة هذه الرحلة مقارنةً بمتوسط جميع الحافلات على "
            "هذا الخط/الاتجاه. المدة المُقارنة تستثني بالفعل التوقف عند طرفي الرحلة (انظر "
            "المعادلة) -- ليست أثرًا ناتجًا عن توقف لم يُحذف."
        ),
        "exp_default": "إشارة استخدمها نموذج الكشف للحكم على أن هذه الرحلة غير طبيعية.",

        "chip_terminus_idle": (
            "⁦{icon}⁩ توقف بالمحطة النهائية: **⁦{min:.0f}⁩ دقيقة** والمحرك/الجهاز نشط قبل "
            "المغادرة أو بعد الوصول (على الأرجح خدمة لم تُغلق — غير محسوبة ضمن مدة الرحلة "
            "أعلاه)."
        ),
        "chip_terminus_idle_help": (
            "هذا الوقت **محذوف** من «مدة الرحلة» المعروضة أعلى البطاقة -- كانت الحافلة لا "
            "تزال ترسل إشارات لكنها لم تبدأ/تنهِ خدمتها فعليًا. إشارة تشغيلية حقيقية (على "
            "الأرجح لم يُغلق السائق الجهاز)، وليست خطأ قياس."
        ),
        "chip_origin_idle": (
            "⁦{icon}⁩ توقف بالمحطة النهائية **⁦{stop}⁩** قبل المغادرة: **⁦{min:.0f}⁩ دقيقة** -- "
            "بلا حركة من ⁦{from_t}⁩ إلى ⁦{to_t}⁩، والمغادرة الفعلية في ⁦{to_t}⁩ (على الأرجح خدمة "
            "لم تُغلق — غير محسوبة ضمن مدة الرحلة أعلاه)."
        ),
        "chip_origin_idle_help": (
            "هذا الوقت **محذوف** من «مدة الرحلة» المعروضة أعلى البطاقة -- كانت الحافلة ترسل "
            "إشارات بالفعل عند محطة المغادرة لكنها لم تبدأ خدمتها بعد. إشارة تشغيلية حقيقية "
            "(على الأرجح لم يُغلق السائق الجهاز)، وليست خطأ قياس."
        ),
        "chip_end_idle": (
            "⁦{icon}⁩ توقف بالمحطة النهائية **⁦{stop}⁩** بعد الوصول: **⁦{min:.0f}⁩ دقيقة** -- "
            "بلا حركة من ⁦{from_t}⁩ إلى ⁦{to_t}⁩ (على الأرجح خدمة لم تُغلق — غير محسوبة ضمن مدة "
            "الرحلة أعلاه)."
        ),
        "chip_end_idle_help": (
            "هذا الوقت **محذوف** من «مدة الرحلة» المعروضة أعلى البطاقة -- كانت الحافلة لا "
            "تزال ترسل إشارات عند محطة الوصول لكنها لم تنهِ خدمتها فعليًا. إشارة تشغيلية "
            "حقيقية (على الأرجح لم يُغلق السائق الجهاز)، وليست خطأ قياس."
        ),
        "chip_real_stop": "⁦{icon}⁩ توقف فعلي: **⁦{stop}⁩** (⁦{min:.0f}⁩ دقيقة بلا حركة ⁦GPS⁩).",
        "chip_real_stop_help": (
            "توقف مرصود أثناء الرحلة نفسها (إشارات ⁦GPS⁩ موجودة، الحافلة متوقفة فعليًا) -- "
            "يبقى محسوبًا ضمن «مدة الرحلة» أعلاه، على عكس توقف المحطة النهائية المحذوف منها."
        ),
        "chip_detour_hint": (
            "&nbsp;&nbsp;↳ *إذا كان هذا التأخير يلي توقف المحطة النهائية أعلاه، فقد لا يكون "
            "نفس المحطة: ربما غادرت الحافلة ثم عادت قبل أن تتوقف من جديد (تحويلة غير رسمية / "
            "مهمة غير مخططة). «عرض خريطة الرحلة» يتحقق من إشارات ⁦GPS⁩ الفعلية ويعرض المسار "
            "الحقيقي إن كان الأمر كذلك.*"
        ),
        "chip_same_terminus_hint": (
            "&nbsp;&nbsp;↳ *هذه هي نفس محطة توقف المحطة النهائية أعلاه (**⁦{stop}⁩**) -- على "
            "الأرجح توقف واحد مستمر انقسم إلى قسمين بسبب انقطاع ⁦GPS⁩ عابر ومعزول، وليس حدثين "
            "منفصلين. المدة الفعلية المرجّحة: ~⁦{total:.0f}⁩ دقيقة.*"
        ),
        "chip_signal_loss": "⁦{icon}⁩ انقطاع إشارة: **⁦{stop}⁩** (~⁦{min:.0f}⁩ دقيقة بلا إشارة).",
        "chip_signal_loss_help": (
            "لم تُستقبل أي إشارة ⁦GPS⁩ لهذه المدة عند هذه المحطة -- قد تكون الحافلة واصلت "
            "السير بشكل طبيعي أثناء هذا الصمت، هذا **ليس** توقفًا مؤكدًا."
        ),
        "chip_dark_gap_between": (
            "⁦{icon}⁩ انقطاع إشارة أثناء الطريق: **بين ⁦{before}⁩ و⁦{after}⁩** (~⁦{min:.0f}⁩ دقيقة "
            "بلا أي إشارة إطلاقًا)."
        ),
        "chip_dark_gap_after_only": (
            "⁦{icon}⁩ انقطاع إشارة أثناء الطريق: **بعد ⁦{before}⁩، لم تُرصد أي محطة أخرى حتى نهاية "
            "الرحلة** (~⁦{min:.0f}⁩ دقيقة بلا أي إشارة إطلاقًا)."
        ),
        "chip_dark_gap_help": (
            "انقطاع إشارة حدث بين محطتين (وليس أثناء الانتظار عند محطة مرصودة بالفعل) -- غير "
            "مرئي في الفحص المعتاد محطة بمحطة، لكنه حقيقي تمامًا: لم يرسل الجهاز أي إشارة "
            "إطلاقًا طوال هذه المدة. هذا وحده يفسّر عادةً ضعف نسبة التتبع وتضخّم مدة بقية "
            "الرحلة -- على الأرجح واصلت الحافلة السير بشكل طبيعي أثناء هذا الصمت."
        ),
        "chip_farthest": "⁦{icon}⁩ محطة مرصودة لكن بموقع مختلف: **⁦{stop}⁩** (~⁦{dist:.0f}⁩ م عن الموقع المتوقع).",
        "chip_farthest_help": (
            "رُصدت الحافلة فعلاً عند هذه المحطة، لكن موقعها عبر ⁦GPS⁩ كان أبعد بشكل ملحوظ عن "
            "الموقع الرسمي للمحطة -- انحراف عن المسار أو انجراف في إشارة ⁦GPS⁩."
        ),
        "chip_off_route": "⁦{icon}⁩ محطات غير مخدومة: ⁦{stops}⁩⁦{suffix}⁩.",
        "chip_off_route_help": (
            "هذه المحطات ضمن نطاق الرحلة لكن لم تُرصد الحافلة أبدًا في مدى ⁦GPS⁩ منها -- رحلة "
            "جزئية، مسار مختلف، أو تخطٍّ فعلي للمحطة."
        ),
        "chip_suspect_coord": (
            "⁦{icon}⁩ ⁦{count}⁩ محطة (محطات) ذات إحداثيات مشكوك فيها (لم تُرصد أبدًا على هذا "
            "الخط — مستبعدة من التشخيص)."
        ),
        "chip_suspect_coord_help": (
            "هذه المحطات **لا تُرصد أبدًا** في أي رحلة على هذا الخط -- إحداثياتها الجغرافية "
            "على الأرجح خاطئة في قاعدة البيانات المرجعية، وليست مشكلة خاصة بهذه الرحلة "
            "تحديدًا. لذلك استُبعدت من التشخيص."
        ),
        "and_others": " (+⁦{n}⁩ أخرى)",

        "btn_show_map": "عرض خريطة الرحلة",
        "btn_hide_map": "إخفاء الخريطة",

        "detour_warning": (
            "**اكتُشفت تحويلة غير رسمية**: غادرت الحافلة نقطة التوقف عند **⁦{left}⁩**، وقطعت "
            "**~⁦{km:.1f}⁩ كم** بعيدًا عنها (أبعد نقطة بلغتها كانت عند **⁦{far}⁩**)، ثم عادت "
            "إليها عند **⁦{back}⁩** (⁦{total:.0f}⁩ دقيقة إجمالاً) — على الأرجح مهمة غير مخططة "
            "أو إعادة تموضع، قبل التوقف الطويل المُبلَّغ عنه أعلاه.⁦{legs}⁩ يُرسم كلا المسارين "
            "الفعليين عبر ⁦GPS⁩ بشكل منفصل أدناه (برتقالي = ذهاب، بنفسجي = عودة) لمتابعة اتجاه "
            "السير (دون الخلط بينهما وبين مسار الخط، المرسوم باللون الرمادي)."
        ),
        "detour_legs": " ذهاب: **⁦{left}⁩ ← ⁦{far}⁩** (⁦{out:.0f}⁩ دقيقة) · عودة: **⁦{far}⁩ ← ⁦{back}⁩** (⁦{back_min:.0f}⁩ دقيقة).",

        "map_no_coords": "إحداثيات ⁦GPS⁩ غير متوفرة لمحطات هذا الخط.",
        "map_hover_time": "وقت المرور: ⁦{v}⁩",
        "map_hover_dwell": "توقف فعلي: ⁦{v}⁩",
        "map_hover_dark": "إشارة مفقودة: ⁦{v}⁩",
        "map_hover_dist": "المسافة عن الموقع المتوقع للمحطة: ⁦{v}⁩",
        "map_hover_tracked": "تتبع ⁦GPS⁩: ⁦{v}⁩",
        "map_hover_yes": "نعم",
        "map_hover_no": "لا",
        "map_hover_suspect": (
            "<br>تنبيه: هذه المحطة لا تُرصد أبدًا على هذا الخط (الإحداثيات على الأرجح خاطئة)"
        ),
        "detour_leg_out_label": "التحويلة · ذهاب",
        "detour_leg_back_label": "التحويلة · عودة",
        "legend_normal_stop": "محطة طبيعية",
        "legend_long_standstill": "توقف طويل",
        "legend_signal_loss": "انقطاع إشارة ⁦GPS⁩",
        "legend_unserved": "محطة غير مخدومة",
        "legend_suspect_coords": "إحداثيات مشكوك فيها",
        "map_legend": (
            "تمثّل كل دائرة محطة، مرقّمة حسب ترتيب المرور (المغادرة ← المحطة النهائية). "
            "يعكس الحجم مدة التوقف + انقطاع الإشارة."
        ),
        "map_normal": "طبيعي",
        "map_long_stop": "توقف طويل (≥10 دقائق)",
        "map_signal_loss": "انقطاع إشارة (≥5 دقائق)",
        "map_unserved": "محطة غير مخدومة (لم تمرّ الحافلة أبدًا)",
        "map_suspect": "إحداثيات محطة مشكوك فيها",
        "map_first_last_tracked": (
            "أول مرور مرصود: **⁦{t0}⁩** (⁦{stop0}⁩) ← آخر مرور مرصود: **⁦{t1}⁩** (⁦{stop1}⁩)"
        ),
        "map_departure": "المغادرة",
        "map_terminus": "المحطة النهائية",
        "map_planned_route": "المسار المخطط",

        "line_good": "خط بحالة جيدة",
        "line_watch": "خط يستدعي المراقبة",
        "line_risk": "خط عالي الخطورة",
        "line_verdict_caption": "الخط ⁦{line}⁩ · ⁦{company}⁩ — استنادًا إلى ⁦{n}⁩ رحلة مُحلَّلة.",
        "metric_anomaly_rate": "معدل الانحراف",
        "metric_flagged_trips": "الرحلات المُبلَّغ عنها",
        "line_not_enough_data": (
            "الخط ⁦{line}⁩: ⁦{n}⁩ رحلة مسجَّلة فقط — بيانات غير كافية للحكم على الحالة العامة "
            "للخط."
        ),

        "ref_trip_expander": "رحلة مرجعية — كيف تبدو الرحلة الطبيعية على الخط ⁦{line}⁩؟",
        "ref_trip_none": "لا توجد رحلة مرجعية متاحة لهذا الخط.",
        "ref_trip_caption": (
            "رحلة حقيقية، اعتبرها النموذج طبيعية، اختيرت من بين الأفضل تتبعًا "
            "(⁦{match:.0f}⁩% من المحطات) بمدة قريبة من متوسط الخط لهذا الاتجاه — قارن حالات "
            "الانحراف أدناه بهذه المرجعية."
        ),
        "ref_trip_bus_day": "الحافلة · اليوم",
        "ref_trip_bus": "الحافلة",
        "ref_trip_duration": "المدة",
        "ref_trip_line_median": "متوسط الخط: ⁦{med}⁩",
        "ref_trip_stops_tracked": "المحطات المرصودة",
        "ref_trip_avg_dwell": "متوسط التوقف",
        "ref_trip_same_cycle": (
            "🔗 دورة كاملة: الحافلة **⁦{bus}⁩**، يوم **⁦{day}⁩** — رحلتا الذهاب والإياب أدناه لنفس "
            "الحافلة ونفس اليوم، لذا فإن التوقف بالمحطة النهائية يعكس استراحة حقيقية بين "
            "الوصول وإعادة الانطلاق."
        ),
        "ref_trip_missing_other_dir": (
            "الاتجاه: **⁦{dir}⁩** (لم يُعثر على رحلة طبيعية صالحة للاتجاه الآخر على هذا الخط)."
        ),
        "ref_typical_idle": (
            "⁦{icon}⁩ التوقف المعتاد بالمحطة النهائية لهذا الاتجاه: **~⁦{typ:.0f}⁩ دقيقة** قبل "
            "المغادرة الفعلية / بعد الوصول الفعلي (متوسط الرحلات الطبيعية). بعد حوالي "
            "**⁦{thr:.0f}⁩ دقيقة** (~ضعف المعدل الطبيعي)، يكون التوقف المرصود على الأرجح "
            "**خدمة لم تُغلق** (لم يُغلق السائق الجهاز على الأرجح) وليس استراحة عادية."
        ),

        "btn_analyze_fail": "تعذّر الحصول على التفسير.",
        "no_anomaly_found": "⁦{scope}⁩: لم يُكتشف أي رحلة غير طبيعية — كل شيء ضمن المعدل الطبيعي.",
        "analysis_header": "⁦{scope}⁩ — تحليل حالات الانحراف",
        "scope_bus_line": "الحافلة ⁦{bus}⁩ · الخط ⁦{line}⁩",
        "scope_line": "الخط ⁦{line}⁩",
        "metric_trips_analyzed": "الرحلات المُحلَّلة",
        "metric_trips_analyzed_help": "العدد الإجمالي للرحلات ضمن الفترة المحددة لهذا النطاق.",
        "metric_abnormal_trips": "الرحلات غير الطبيعية",
        "metric_abnormal_trips_help": "الرحلات التي أشار إليها نموذج الكشف بأنها غير طبيعية (⁦Isolation Forest⁩ + ⁦LSTM⁩).",
        "metric_normal_duration": "المدة الطبيعية (الوسيط)",
        "metric_normal_duration_help": "الوسيط الزمني لرحلة طبيعية على هذا الخط -- يُستخدم كمرجع للحكم إن كانت رحلة ما طويلة أو قصيرة بشكل غير معتاد.",
        "section_flagged_trips": "الرحلات المُبلَّغ عنها",
        "section_trip_detail": "تحليل مفصّل لرحلة",
        "no_trip_matches_filter": "لا توجد رحلة مطابقة للفلتر المحدد أعلاه.",
        "select_trip_prompt": "اختر رحلة أدناه لعرض خريطة محطاتها، ورسم بياني للتوقفات، والجدول الكامل محطة بمحطة.",
        "trip_to_analyze": "الرحلة المراد تحليلها:",
        "metric_delta_vs_normal": "الفارق عن المعدل الطبيعي",
        "metric_delta_vs_normal_help": "الفرق بين هذه الرحلة والمدة الطبيعية للخط. فارق موجب يعني رحلة أطول من المعتاد.",
        "metric_stops_served": "المحطات المخدومة",
        "metric_stops_served_help": "عدد المحطات التي رُصدت فيها الحافلة ضمن مدى ⁦GPS⁩، من إجمالي المحطات المخطط لها على الخط.",
        "metric_anomaly_score": "درجة الانحراف",
        "metric_anomaly_score_help": "الدرجة المحسوبة بواسطة نموذج ⁦Isolation Forest⁩. كلما ارتفعت، زاد ابتعاد سلوك الحافلة عن المعتاد.",
        "section_trip_map": "خريطة الرحلة",
        "section_dwell_chart": "التوقف وانقطاع الإشارة حسب المحطة",
        "dwell_chart_caption": (
            "الأعمدة الزرقاء تُظهر الوقت الفعلي الذي كانت فيه الحافلة متوقفة و⁦GPS⁩ نشط. الأعمدة "
            "الصفراء تُظهر فترات بلا إشارة ⁦GPS⁩ عند تلك المحطة (إشارة مفقودة). الأعمدة الحمراء "
            "تقابل المحطات غير المخدومة."
        ),
        "series_real_standstill": "توقف فعلي (⁦GPS⁩ نشط)",
        "series_signal_lost": "إشارة مفقودة (بلا نبضة)",
        "hover_real_standstill": "<b>%⁦{x}⁩</b><br>توقف فعلي: %⁦{y:.1f}⁩ دقيقة<extra></extra>",
        "hover_signal_lost": "<b>%⁦{x}⁩</b><br>إشارة مفقودة: %⁦{y:.1f}⁩ دقيقة<extra></extra>",
        "axis_minutes": "دقائق",
        "section_stop_table": "جدول محطة بمحطة",
        "stop_table_caption": (
            "**تتبع ⁦GPS⁩**: مرّت الحافلة بمنطقة المحطة وتم رصدها. **توقف فعلي**: مدة التوقف "
            "و⁦GPS⁩ نشط. **إشارة مفقودة**: مدة بلا نبضة ⁦GPS⁩ عند هذه المحطة (غير محسوبة كتوقف). "
            "**مسافة المحطة**: الفارق بين موقع ⁦GPS⁩ للحافلة والموقع الرسمي للمحطة."
        ),
        "col_stop": "المحطة", "col_gps_tracked": "تتبع ⁦GPS⁩", "col_real_standstill": "توقف فعلي",
        "col_signal_lost": "إشارة مفقودة", "col_stop_distance": "مسافة المحطة",
        "val_tracked": "مرصودة", "val_unserved": "غير مخدومة",
        "no_sequence_data": "لا توجد بيانات تسلسل متاحة لهذه الرحلة.",
    },
}


def get_lang() -> str:
    return st.session_state.get("lang", "fr")


def set_lang(lang: str) -> None:
    st.session_state["lang"] = lang


def t(key: str, **kwargs) -> str:
    lang = get_lang()
    template = TRANSLATIONS.get(lang, {}).get(key)
    if template is None:
        template = TRANSLATIONS["fr"].get(key, key)
    if not kwargs:
        return template
    try:
        return template.format(**kwargs)
    except Exception:
        return template
