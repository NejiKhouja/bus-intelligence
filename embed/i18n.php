<?php
/**
 * Traductions des quelques chaînes rendues côté PHP (index.php) -- tout le reste de
 * l'interface (app.js) est traduit séparément par assets/i18n.js, ce fichier ne couvre
 * QUE ce que PHP écrit directement dans le HTML avant que JS ne s'exécute (titre de
 * page, en-tête, onglets, écran de sélection d'opérateur). Même schéma minimal que
 * src/dashboard/i18n.py côté Streamlit : un dict {langue: {clé: texte}}, retombe sur le
 * français si la clé manque en arabe.
 */

const WINICARI_I18N = [
    'fr' => [
        'page_title'        => 'WiniCari AI — Détection d\'anomalies',
        'header_title'      => 'Détection d\'anomalies',
        'header_subtitle'   => 'Suivi automatique des trajets et alertes en cas de comportement inhabituel',
        'change_company'    => 'changer',
        'tab_trips'         => 'Trajets signalés',
        'tab_trends'        => 'Tendances',
        'tab_tickets'       => 'Anomalies billetterie',
        'tab_drivers'       => 'Chauffeurs',
        'tab_traceline'     => 'Tracer une nouvelle ligne',
        'picker_heading'    => 'Choisir un opérateur',
        'picker_body'       => 'Aucun opérateur n\'est associé à cette session. Dans l\'intégration finale, votre système de connexion définira <code>$_SESSION[\'winicari_company\']</code> automatiquement — ce choix manuel n\'est là que pour le développement/test.',
        'picker_select'     => 'Sélectionner…',
        'picker_continue'   => 'Continuer',
        'lang_label'        => 'Langue',
        'footer_disclaimer' => 'Les informations, statistiques et explications affichées ci-dessus sont générées automatiquement par un système d\'intelligence artificielle (détection d\'anomalies, analyse de billetterie) et peuvent contenir des erreurs, des approximations ou des interprétations incorrectes. Elles sont fournies à titre indicatif uniquement et ne constituent pas un verdict définitif sur un trajet, un chauffeur ou un opérateur. WiniCari AI décline toute responsabilité quant aux décisions prises sur la base de ces informations : il revient à chaque utilisateur de les vérifier et de les interpréter selon son propre jugement et le contexte.',
    ],
    'ar' => [
        'page_title'        => 'ويني كاري AI — كشف الخلل',
        'header_title'      => 'كشف الخلل',
        'header_subtitle'   => 'متابعة آلية للرحلات وتنبيهات عند رصد سلوك غير معتاد',
        'change_company'    => 'تغيير',
        'tab_trips'         => 'الرحلات المُبلّغ عنها',
        'tab_trends'        => 'الاتجاهات',
        'tab_tickets'       => 'خلل التذاكر',
        'tab_drivers'       => 'السائقون',
        'tab_traceline'     => 'رسم خط جديد',
        'picker_heading'    => 'اختيار المشغّل',
        'picker_body'       => 'لا يوجد مشغّل مرتبط بهذه الجلسة. في التكامل النهائي، سيحدد نظام تسجيل الدخول الخاص بكم <code>$_SESSION[\'winicari_company\']</code> تلقائيًا — هذا الاختيار اليدوي مخصص فقط للتطوير/الاختبار.',
        'picker_select'     => 'اختر…',
        'picker_continue'   => 'متابعة',
        'lang_label'        => 'اللغة',
        'footer_disclaimer' => 'المعلومات والإحصائيات والتفسيرات المعروضة أعلاه يتم توليدها آليًا بواسطة نظام ذكاء اصطناعي (كشف الخلل، تحليل بيانات التذاكر) وقد تحتوي على أخطاء أو تقريبات أو تأويلات غير دقيقة. هي مُقدَّمة على سبيل الاستئناس فقط ولا تُعتبر حكمًا نهائيًا بخصوص رحلة أو سائق أو مشغّل. تُخلي ويني كاري AI مسؤوليتها عن أي قرار يُتخذ استنادًا إلى هذه المعلومات: يعود لكل مستخدم التحقق منها وتفسيرها وفق تقديره الخاص وسياق كل حالة.',
    ],
];

function wt(string $key): string {
    $lang = function_exists('winicari_current_lang') ? winicari_current_lang() : 'fr';
    return WINICARI_I18N[$lang][$key] ?? WINICARI_I18N['fr'][$key] ?? $key;
}
