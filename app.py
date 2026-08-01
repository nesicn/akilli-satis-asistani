import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import random
from datetime import datetime

# ==========================================
# 1. SAYFA YAPILANDIRMASI & TEMA
# ==========================================
st.set_page_config(
    page_title="Smart Checkout AI - Kasa Asistanı",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .xai-box {
        background-color: #F0FDF4;
        border-left: 4px solid #16A34A;
        padding: 12px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .xai-box-low {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 12px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .xai-box-neutral {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 12px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL ŞEMASI SABİTLERİ (smart_kasa_model.pkl ile birebir uyumlu)
# ==========================================
# 'Onerilen_Kategori' görünen adı <-> sütun öneki (örn. "Cilt Bakımı" -> "Cilt_Bakimi").
# Model 10 kategori üzerinde eğitildi; DB şeması, XAI ve ürün önerisi TEK bu
# kaynaktan besleniyor (tutarlılık için).
KATEGORI_KOLON_ONEKI = {
    "Makyaj": "Makyaj",
    "Cilt Bakımı": "Cilt_Bakimi",
    "Ağız Bakımı": "Agiz_Bakimi",
    "Parfüm": "Parfum",
    "Saç Bakımı": "Sac_Bakimi",
    "Vücut Bakımı & Banyo": "Vucut_Bakimi_Banyo",
    "El & Ayak Bakımı": "El_Ayak_Bakimi",
    "Güneş & Bronzlaşma": "Gunes_Bronzlasma",
    "Aksesuar & Güzellik Aletleri": "Aksesuar_Guzellik_Aletleri",
    "Erkek Bakım": "Erkek_Bakim",
}

KATEGORI_URUN_ONERI = {
    "Makyaj": ("Göz Farı", 79.90),
    "Cilt Bakımı": ("Misel Su", 69.90),
    "Ağız Bakımı": ("Ağız Çalkalama Suyu", 44.90),
    "Parfüm": ("Vücut Spreyi (Body Mist)", 59.90),
    "Saç Bakımı": ("Kuru Şampuan", 64.90),
    "Vücut Bakımı & Banyo": ("Banyo Bombası", 29.90),
    "El & Ayak Bakımı": ("Aseton", 24.90),
    "Güneş & Bronzlaşma": ("Aloe Vera Jeli", 39.90),
    "Aksesuar & Güzellik Aletleri": ("Gua Sha Taşı", 89.90),
    "Erkek Bakım": ("Tıraş Köpüğü", 54.90),
}

# Kategorik özellikler. NOT: Hafta_Ici_Hafta_Sonu veride 0/1 TAM SAYI olarak
# saklanıyor ve model TAM SAYI (int) kategorileriyle eğitildi (0, 1) — metne
# çevrilirse (örn. "Hafta Sonu") OneHotEncoder bunu "bilinmeyen kategori"
# sayıp sessizce sıfırlar. Bu yüzden DB'de de INTEGER olarak tutuluyor.
MUSTERI_KATEGORIK_KOLONLAR = [
    "Cinsiyet", "Yas_Grubu", "Magaza_Tipi", "Mevsim",
    "Islem_Saati_Dilimi", "Hafta_Ici_Hafta_Sonu",
]
_KATEGORIK_SQL_TIPI = {"Hafta_Ici_Hafta_Sonu": "INTEGER"}  # diğerleri TEXT

MUSTERI_GENEL_SAYISAL_KOLONLAR = [
    "Sadakat_Puani", "Promosyon_Hassasiyeti_Skoru", "Gecen_Gun_Sayisi",
    "Coklu_Kategori_Alim_Skoru", "Sepet_Tutari_TL",
]

MUSTERI_KATEGORI_SAYISAL_KOLONLAR = []
for _onek in KATEGORI_KOLON_ONEKI.values():
    MUSTERI_KATEGORI_SAYISAL_KOLONLAR += [
        f"{_onek}_Alisveris_Sayisi", f"Gecen_Gun_{_onek}",
        f"{_onek}_Ort_Alim_Araligi", f"{_onek}_Tuketim_Orani",
    ]

MUSTERI_TUM_SAYISAL_KOLONLAR = MUSTERI_GENEL_SAYISAL_KOLONLAR + MUSTERI_KATEGORI_SAYISAL_KOLONLAR
# DB'deki TÜM müşteri öznitelik sütunları (kimlik hariç) - modelin ihtiyaç
# duyduğu her şeyi kapsar; init_database() bu listeyi eksiksiz oluşturur.
MUSTERI_TUM_OZNITELIK_KOLONLARI = MUSTERI_KATEGORIK_KOLONLAR + MUSTERI_TUM_SAYISAL_KOLONLAR

# ==========================================
# 3. VERİTABANI (SQLITE) MİMARİSİ VE YÖNETİMİ
# ==========================================
DB_FILE = "kasa_veritabani.db"
MUSTERI_CSV_YOLU = "musteri_davranis_seti.csv"


def get_db_connection():
    """SQLite veritabanı bağlantısı oluşturur."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def _musteriler_create_table_sql():
    satirlar = ["Musteri_ID TEXT PRIMARY KEY", "telefon TEXT UNIQUE", "Ad_Soyad TEXT"]
    for kolon in MUSTERI_KATEGORIK_KOLONLAR:
        satirlar.append(f"{kolon} {_KATEGORIK_SQL_TIPI.get(kolon, 'TEXT')}")
    for kolon in MUSTERI_TUM_SAYISAL_KOLONLAR:
        satirlar.append(f"{kolon} REAL")
    return "CREATE TABLE musteriler (\n    " + ",\n    ".join(satirlar) + "\n)"


def init_database():
    """Veritabanını başlatır/gerekirse şemayı onarır ve CSV'deki verileri aktarır.

    Not: Tablo daha önce (modelin ihtiyaç duyduğu tüm sütunları içermeyen) DAR
    bir şemayla oluşturulmuşsa, "columns are missing" hatasını kalıcı olarak
    çözmek için tablo güvenli şekilde yeniden oluşturulur (DDL komutlarına
    kullanıcı girdisi hiçbir zaman karışmaz).
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='musteriler'")
    tablo_var_mi = cursor.fetchone() is not None
    if tablo_var_mi:
        cursor.execute("PRAGMA table_info(musteriler)")
        mevcut_kolonlar = {row[1] for row in cursor.fetchall()}
        beklenen = set(MUSTERI_TUM_OZNITELIK_KOLONLARI) | {"Musteri_ID", "telefon", "Ad_Soyad"}
        if not beklenen.issubset(mevcut_kolonlar):
            cursor.execute("DROP TABLE musteriler")
            conn.commit()
            tablo_var_mi = False

    if not tablo_var_mi:
        cursor.execute(_musteriler_create_table_sql())

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS oneri_gecmisi (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            musteri_id TEXT,
            ad_soyad TEXT,
            kategori TEXT,
            urun TEXT,
            tutar REAL,
            olasilik REAL,
            sonuc TEXT,
            zaman TEXT
        )
    """)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM musteriler")
    count = cursor.fetchone()[0]

    if count == 0 and os.path.exists(MUSTERI_CSV_YOLU):
        try:
            df_csv = pd.read_csv(MUSTERI_CSV_YOLU, encoding="utf-8-sig", dtype={"Musteri_ID": str})
            df_csv.columns = [str(c).strip() for c in df_csv.columns]

            aktarilacak_kolonlar = ["Musteri_ID", "Ad_Soyad"] + [c for c in MUSTERI_TUM_OZNITELIK_KOLONLARI if c in df_csv.columns]
            df_to_save = df_csv[aktarilacak_kolonlar].copy()
            # CSV'de telefon numarası yok (bu satır set'i toplu/tarihsel müşterileri
            # temsil ediyor); telefon sütunu bu müşteriler için NULL kalır ve
            # onlar Musteri_ID üzerinden aranabilir.
            df_to_save.to_sql("musteriler", conn, if_exists="append", index=False)
            conn.commit()
        except Exception as e:
            st.error(f"CSV'den SQLite'a veri aktarılırken hata: {e}")

    conn.close()


init_database()


def build_default_profile_from_db():
    """Kayıtlı/eşleşen müşteri olmadığında modele SIFIRLAR yerine bu genel
    (ortalama/en sık değer) profili veriyoruz; aksi halde tahmin sessizce
    anlamsızlaşır. Değerler doğrudan veritabanından SQL agregasyonlarıyla
    hesaplanır."""
    conn = get_db_connection()
    cursor = conn.cursor()
    profil = {}
    try:
        for kolon in MUSTERI_TUM_SAYISAL_KOLONLAR:
            cursor.execute(f"SELECT AVG({kolon}) FROM musteriler WHERE {kolon} IS NOT NULL")
            deger = cursor.fetchone()[0]
            profil[kolon] = float(deger) if deger is not None else 0.0
        for kolon in MUSTERI_KATEGORIK_KOLONLAR:
            cursor.execute(f"""
                SELECT {kolon}, COUNT(*) c FROM musteriler
                WHERE {kolon} IS NOT NULL GROUP BY {kolon} ORDER BY c DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row is not None:
                profil[kolon] = row[0]
            else:
                profil[kolon] = 0 if kolon == "Hafta_Ici_Hafta_Sonu" else "Bilinmiyor"
    finally:
        conn.close()
    return profil


def _sadakat_esik_degerleri():
    """VIP / Sadık Müşteri ayrımı için Sadakat_Puani dağılımından eşikleri
    hesaplar (üst %25 / medyan). SQLite'ta yerleşik yüzdelik fonksiyonu
    olmadığından değerler çekilip numpy ile hesaplanır (5000 satır için
    önemsiz maliyetli)."""
    conn = get_db_connection()
    try:
        puanlar = pd.read_sql_query("SELECT Sadakat_Puani FROM musteriler WHERE Sadakat_Puani IS NOT NULL", conn)["Sadakat_Puani"]
    finally:
        conn.close()
    if puanlar.empty:
        return 0.0, 0.0
    return float(puanlar.quantile(0.75)), float(puanlar.median())


def _sik_alisveris_esigi():
    """XAI'deki 'sık alışveriş yapan müşteri' içgörüsü için eşik: tüm
    kategorilerdeki alışveriş sayıları toplamının üst %25'i."""
    conn = get_db_connection()
    try:
        kolonlar = [f"{onek}_Alisveris_Sayisi" for onek in KATEGORI_KOLON_ONEKI.values()]
        toplam_ifadesi = " + ".join([f"COALESCE({k}, 0)" for k in kolonlar])
        df = pd.read_sql_query(f"SELECT ({toplam_ifadesi}) AS toplam FROM musteriler", conn)["toplam"]
    finally:
        conn.close()
    return float(df.quantile(0.75)) if not df.empty else 15.0


def musteri_ara(arama_terimi):
    """Telefon numarası VEYA Musteri_ID ile güvenli (parameterized) arama.
    Her iki değer de '?' yer tutucusu ile bağlanır; ham girdi asla SQL
    metnine dahil edilmez (SQL injection'a karşı güvenli)."""
    if not arama_terimi:
        return None
    terim = arama_terimi.strip()
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM musteriler WHERE telefon = ? OR Musteri_ID = ? LIMIT 1", (terim, terim))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def yeni_musteri_ekle(ad_soyad, telefon, segment, cinsiyet, yas_grubu, magaza_tipi, mevsim, genel_profil):
    """Güvenli (parameterized) müşteri ekleme. Formda toplanmayan tüm
    öznitelikler nüfus-geneli varsayılan profille doldurulur; böylece yeni
    müşteri de ilk andan itibaren modelin tüm sütunlarına sahip olur."""
    puan_map = {"Yeni Müşteri": 10.0, "Standart Müşteri": 30.0, "Sadık Müşteri": 65.0, "VIP Müşteri": 90.0}
    puan = puan_map.get(segment, 20.0)
    yeni_id = f"MST-N{random.randint(10000, 99999)}"

    # Kayıt sözlüğü: anahtarlar HER ZAMAN kod-içi sabit listelerden
    # (MUSTERI_TUM_OZNITELIK_KOLONLARI / genel_profil) geldiği için sütun
    # adları asla kullanıcı girdisinden türetilmez. Sadece DEĞERLER
    # (ad_soyad, telefon gibi) kullanıcıdan gelir ve '?' ile parametrize edilir.
    kayit = dict(genel_profil)
    kayit.update({
        "Cinsiyet": cinsiyet, "Yas_Grubu": yas_grubu,
        "Magaza_Tipi": magaza_tipi, "Mevsim": mevsim,
        "Sadakat_Puani": puan, "Sepet_Tutari_TL": 0.0, "Gecen_Gun_Sayisi": 0,
    })
    kayit["Hafta_Ici_Hafta_Sonu"] = int(kayit.get("Hafta_Ici_Hafta_Sonu", 0) or 0)

    kolonlar = ["Musteri_ID", "telefon", "Ad_Soyad"] + [k for k in MUSTERI_TUM_OZNITELIK_KOLONLARI if k in kayit]
    degerler = [yeni_id, telefon.strip(), ad_soyad.strip()] + [kayit[k] for k in kolonlar[3:]]
    yer_tutucular = ", ".join(["?"] * len(kolonlar))

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute(f"INSERT INTO musteriler ({', '.join(kolonlar)}) VALUES ({yer_tutucular})", degerler)
        conn.commit()
        return True, "Müşteri veritabanına başarıyla eklendi!", yeni_id
    except sqlite3.IntegrityError:
        return False, "Bu telefon numarasıyla kayıtlı bir müşteri zaten var!", None
    except Exception as e:
        return False, f"Hata oluştu: {str(e)}", None
    finally:
        conn.close()


def musteri_satis_kaydet(musteri_kimlik, harcanan_tutar, secilen_kategori):
    """Satış tamamlandığında müşterinin genel VE kategori-bazlı geçmişini
    canlı günceller (sonraki ziyarette daha isabetli tahmin için)."""
    if not musteri_kimlik:
        return False
    kimlik = musteri_kimlik.strip()
    onek = KATEGORI_KOLON_ONEKI.get(secilen_kategori)  # sabit, kod-içi eşlemeden -> güvenli

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if onek:
            kolon_gecen_gun = f"Gecen_Gun_{onek}"
            kolon_alisveris = f"{onek}_Alisveris_Sayisi"
            cursor.execute(f"""
                UPDATE musteriler
                SET Sepet_Tutari_TL = ?,
                    Gecen_Gun_Sayisi = 0,
                    {kolon_gecen_gun} = 0,
                    {kolon_alisveris} = COALESCE({kolon_alisveris}, 0) + 1
                WHERE telefon = ? OR Musteri_ID = ?
            """, (harcanan_tutar, kimlik, kimlik))
        else:
            cursor.execute("""
                UPDATE musteriler SET Sepet_Tutari_TL = ?, Gecen_Gun_Sayisi = 0
                WHERE telefon = ? OR Musteri_ID = ?
            """, (harcanan_tutar, kimlik, kimlik))
        conn.commit()
        return True
    except Exception:
        return False
    finally:
        conn.close()


def oneri_kaydet(musteri_id, ad_soyad, kategori, urun, tutar, olasilik, sonuc):
    """Her AI önerisini (kabul/red) kalıcı geçmişe yazar -> Tab 2'deki
    'Son Öneriler' tablosunu ve ROI metriklerini besler."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO oneri_gecmisi (musteri_id, ad_soyad, kategori, urun, tutar, olasilik, sonuc, zaman)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (musteri_id, ad_soyad, kategori, urun, tutar, olasilik, sonuc, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def son_onerileri_getir(limit=8):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM oneri_gecmisi ORDER BY id DESC LIMIT ?", (limit,))
    satirlar = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return satirlar


# ==========================================
# 4. SESSION STATE İLKLEME
# ==========================================
if "total_recommendations" not in st.session_state:
    st.session_state.total_recommendations = 0
if "accepted_recommendations" not in st.session_state:
    st.session_state.accepted_recommendations = 0
if "rejected_recommendations" not in st.session_state:
    st.session_state.rejected_recommendations = 0
if "ai_generated_revenue" not in st.session_state:
    st.session_state.ai_generated_revenue = 0.0
if "analiz_yapildi" not in st.session_state:
    st.session_state.analiz_yapildi = False
if "last_feedback_msg" not in st.session_state:
    st.session_state.last_feedback_msg = None
if "last_feedback_type" not in st.session_state:
    st.session_state.last_feedback_type = None


def reset_analysis():
    st.session_state.analiz_yapildi = False
    st.session_state.last_feedback_msg = None
    st.session_state.last_feedback_type = None


# ==========================================
# 5. MODEL YÜKLEME
# ==========================================
MODEL_YOLU = "smart_kasa_model.pkl"


@st.cache_resource
def load_model(model_path=MODEL_YOLU):
    """Yapay zeka modelini güvenli şekilde yükler."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı: `{model_path}`.")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata oluştu: {str(e)}")
        return None


model = load_model()
GENEL_MUSTERI_PROFILI = build_default_profile_from_db()
VIP_ESIK_DEGERI, SADIK_ESIK_DEGERI = _sadakat_esik_degerleri()
SIK_ALISVERIS_ESIGI = _sik_alisveris_esigi()


def musteri_segment_belirle(sadakat_puani):
    try:
        puan = float(sadakat_puani)
    except (TypeError, ValueError):
        return "Standart"
    if puan >= VIP_ESIK_DEGERI:
        return "VIP"
    if puan >= SADIK_ESIK_DEGERI:
        return "Sadık Müşteri"
    return "Standart"


def mask_text(text, visible_chars=2):
    if not isinstance(text, str) or len(text) <= visible_chars:
        return text
    return text[:visible_chars] + "*" * (len(text) - visible_chars)


def generate_xai_insights(secili_musteri, harcanan_tutar, secilen_kategori, proba):
    """Zengin, çok sinyalli açıklanabilirlik (XAI) kutuları üretir."""
    insights = []

    sadakat_puani = secili_musteri.get('Sadakat_Puani') if secili_musteri else None
    segment = musteri_segment_belirle(sadakat_puani) if sadakat_puani is not None else 'Standart'
    if segment == 'VIP':
        insights.append(("pos", f"⭐ **VIP Müşteri Sadakati:** Sadakat puanı ({sadakat_puani:.0f}) üst %25'te — yüksek bağlılık ikna olasılığını artırıyor."))
    elif segment == 'Sadık Müşteri':
        insights.append(("pos", f"💙 **Sadık Müşteri Profili:** Sadakat puanı ({sadakat_puani:.0f}) medyanın üzerinde — düzenli ziyaretler öneri kabul esnekliğini destekliyor."))
    elif sadakat_puani is not None:
        insights.append(("neu", f"📉 **Gelişmekte Olan Sadakat:** Sadakat puanı ({sadakat_puani:.0f}) henüz düşük — teklif daha temkinli değerlendirilebilir."))

    kat_onek = KATEGORI_KOLON_ONEKI.get(secilen_kategori)
    if secili_musteri and kat_onek:
        gecen_gun = float(secili_musteri.get(f"Gecen_Gun_{kat_onek}", 30) or 30)
        ort_aralik = float(secili_musteri.get(f"{kat_onek}_Ort_Alim_Araligi", 30) or 30)
        tuketim_orani = float(secili_musteri.get(f"{kat_onek}_Tuketim_Orani", 0.5) or 0.5)
        alisveris_sayisi = float(secili_musteri.get(f"{kat_onek}_Alisveris_Sayisi", 0) or 0)

        if ort_aralik > 0 and gecen_gun >= ort_aralik:
            insights.append(("pos", f"⏳ **Yenileme Zamanı Gelmiş ({secilen_kategori}):** Son alımdan bu yana {int(gecen_gun)} gün geçmiş (Ort. Döngü: {int(ort_aralik)} gün). Müşterinin bu ürüne ihtiyacı yüksek."))
        elif ort_aralik > 0:
            insights.append(("neu", f"🕐 **Henüz Erken ({secilen_kategori}):** Son alımdan {int(gecen_gun)} gün geçmiş, ortalama döngüsü {int(ort_aralik)} gün — henüz vaktinden önce olabilir."))

        if tuketim_orani > 0.6:
            insights.append(("pos", f"📈 **Yüksek Tüketim Skoru:** {secilen_kategori} kategorisindeki geçmiş tüketim oranı (%{tuketim_orani*100:.0f}) yüksek."))

        if alisveris_sayisi == 0:
            insights.append(("neu", f"🆕 **İlk Kez ({secilen_kategori}):** Müşterinin bu kategoride geçmiş alışverişi yok; öneri keşif amaçlı olabilir."))

    if harcanan_tutar >= 500:
        insights.append(("pos", f"💰 **Yüksek Alışveriş Hacmi:** {harcanan_tutar:.0f} TL tutarındaki sepet, müşterinin ek tekliflere açık olduğunu gösteriyor."))
    elif harcanan_tutar < 200:
        insights.append(("neg", "⚠️ **Düşük Sepet Tutarı:** Müşteri hassas bir bütçeyle alışveriş yapıyor olabilir."))

    if secili_musteri:
        toplam_alisveris = sum(float(secili_musteri.get(f"{onek}_Alisveris_Sayisi", 0) or 0) for onek in KATEGORI_KOLON_ONEKI.values())
        if toplam_alisveris > SIK_ALISVERIS_ESIGI:
            insights.append(("pos", f"🔄 **Sık Alışveriş Yapan Müşteri:** Geçmişteki {int(toplam_alisveris)} kategori bazlı alışveriş güven indeksini yükseltiyor."))

        promosyon_skoru = secili_musteri.get('Promosyon_Hassasiyeti_Skoru')
        if promosyon_skoru is not None and float(promosyon_skoru) > 0.6:
            insights.append(("pos", f"🏷️ **Promosyona Duyarlı:** Promosyon hassasiyet skoru (%{float(promosyon_skoru)*100:.0f}) yüksek — fırsat vurgusu etkili olabilir."))
    else:
        insights.append(("neu", "ℹ️ Kayıtlı profil bulunamadı; genel popülasyon istatistikleri (medyan/en sık değer) esas alındı."))

    if not insights:
        insights.append(("neu", "ℹ️ Genel sepet ortalamaları ve mağaza içi standart müşteri profil davranışları esas alındı."))

    return insights


# ==========================================
# 6. YAN MENÜ (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/checkout.png", width=70)
    st.title("Smart Checkout AI")
    st.caption("SQLite & AI Powered POS")
    st.divider()

    maskeleme_aktif = st.toggle("🔒 Maskeleme Modu", value=False)
    st.info("Sistem, kasa anında akıllı çapraz satış ve müşteri ikna olasılığı tahmini üretir.")

    st.divider()
    st.markdown("**Sistem Durumu:**")
    if model is not None:
        _model_nesnesi = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
        _model_adi = type(_model_nesnesi).__name__.replace("Classifier", "")
        st.success(f"🟢 AI Modeli & SQLite Aktif (`{_model_adi}`)")
    else:
        st.warning("🟠 Model Bulunamadı (Simülasyon Modu)")

    st.divider()
    st.markdown("**📈 Canlı ROI Metrikleri:**")
    st.metric("Ek Ciro Katkısı", f"{st.session_state.ai_generated_revenue:.2f} TL")
    acc_rate = (st.session_state.accepted_recommendations / st.session_state.total_recommendations * 100) if st.session_state.total_recommendations > 0 else 0.0
    st.metric("Kasa Dönüşüm Oranı", f"%{acc_rate:.1f}")

# ==========================================
# 7. ANA EKRAN VE SEKMELER
# ==========================================
st.title("🛒 Akıllı Kasa Asistanı & Öneri Motoru")

tab_kasa, tab_analitik, tab_yeni_musteri = st.tabs([
    "🛍️ Kasa İşlem Ekranı",
    "📊 Mağaza & ROI Analitiği",
    "➕ Yeni Müşteri Kaydı"
])

# ------------------------------------------
# TAB 1: KASA İŞLEM EKRANI
# ------------------------------------------
with tab_kasa:
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("1. Müşteri & Sepet Bilgileri")

        musteri_kimlik_input = st.text_input(
            "📱 Müşteri Telefon No veya 🆔 Müşteri ID:",
            placeholder="05xxxxxxxxx  veya  MST-00001",
            key="input_musteri_kimlik",
            on_change=reset_analysis
        )

        secili_musteri = musteri_ara(musteri_kimlik_input) if musteri_kimlik_input else None
        if musteri_kimlik_input:
            if secili_musteri:
                ad_display = mask_text(secili_musteri.get('Ad_Soyad', 'Bilinmeyen Müşteri')) if maskeleme_aktif else secili_musteri.get('Ad_Soyad', 'Bilinmeyen Müşteri')
                musteri_segmenti = musteri_segment_belirle(secili_musteri.get('Sadakat_Puani'))
                st.success(f"👤 Müşteri Bulundu: **{ad_display}** ({musteri_segmenti} Segment)")
            else:
                st.info("ℹ️ Kayıtlı müşteri bulunamadı. Genel müşteri profili ile devam ediliyor.")

        st.divider()

        harcanan_tutar = st.number_input(
            "💰 Anlık Sepet Tutarı (TL):", min_value=10.0, max_value=10000.0,
            value=350.0, step=10.0, key="input_tutar", on_change=reset_analysis
        )
        sepetteki_urun = st.slider(
            "📦 Sepetteki Ürün Adedi:", min_value=1, max_value=20, value=3,
            key="input_urun_adedi", on_change=reset_analysis
        )
        secilen_kategori = st.selectbox(
            "🏷️ Sepetteki Ağırlıklı Kategori:", list(KATEGORI_KOLON_ONEKI.keys()),
            key="input_kat", on_change=reset_analysis
        )

        if st.button("⚡ AI Önerilerini ve Tahmini Hesapla", type="primary", use_container_width=True):
            st.session_state.analiz_yapildi = True
            st.session_state.last_feedback_msg = None
            st.session_state.last_feedback_type = None

    with col_right:
        st.subheader("2. Yapay Zeka Karar Destek")

        if st.session_state.analiz_yapildi:
            if model is None:
                st.error("Model yüklü olmadığı için tahmin yapılamıyor.")
            else:
                oneri_urun, oneri_fiyat = KATEGORI_URUN_ONERI.get(secilen_kategori, ("Kasa Önü Minis Ürünler", 19.90))

                # --- A. FEATURE VEKTÖRÜNÜ HAZIRLAMA (53 ÖZELLİKLİ MODEL MİMARİSİ) ---
                # SQLite'tan gelen müşteri satırı ARTIK modelin ihtiyaç duyduğu tüm
                # sütunları içeriyor (bkz. MUSTERI_TUM_OZNITELIK_KOLONLARI). Eşleşen
                # müşteri yoksa nüfus-geneli (medyan/mod) profille başlanır. Son
                # savunma hattı olarak reindex + fill_value=0 kalır: hangi kaynaktan
                # gelirse gelsin, "columns are missing" hatası artık YAPISAL olarak
                # imkansızdır.
                input_dict = dict(secili_musteri) if secili_musteri else dict(GENEL_MUSTERI_PROFILI)
                input_dict["Sepet_Tutari_TL"] = harcanan_tutar
                input_dict["Onerilen_Kategori"] = secilen_kategori
                input_dict["Onerilen_Urun"] = oneri_urun

                df_input = pd.DataFrame([input_dict])
                df_input = df_input.where(pd.notnull(df_input), None)  # SQL NULL -> NaN uyumu
                expected_features = getattr(model, "feature_names_in_", None)
                if expected_features is not None:
                    df_input = df_input.reindex(columns=list(expected_features), fill_value=0)

                # --- B. AI TAHMİNİ ---
                try:
                    if hasattr(model, "predict_proba"):
                        proba = float(model.predict_proba(df_input)[0][1])
                    else:
                        pred = model.predict(df_input)[0]
                        proba = 0.85 if pred == 1 else 0.35
                except Exception as e:
                    st.warning(f"Model tahmini sırasında uyarı oluştu, yaklaşık skor kullanılıyor: {str(e)}")
                    proba = min(0.92, max(0.25, (harcanan_tutar / 1000.0) * 0.5 + (0.3 if secili_musteri else 0.1)))

                st.markdown("#### 🎯 Müşteri Ek Satın Alma İkna Olasılığı")
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric(label="Tahmin Skoru", value=f"%{proba * 100:.1f}")
                with c2:
                    st.progress(float(proba))
                    if proba >= 0.70:
                        st.caption("🔥 **Yüksek Potansiyel:** Müşteri kasada ek önerilere çok açık.")
                    elif proba >= 0.40:
                        st.caption("🟡 **Orta Potansiyel:** Standart kasa fırsat ürünleri sunulabilir.")
                    else:
                        st.caption("⚪ **Düşük Potansiyel:** Hızlı işlem tamamlanması önerilir.")

                st.divider()

                # --- C. XAI (AÇIKLANABİLİR AI) ---
                st.markdown("#### 🔍 Yapay Zeka Karar Gerekçeleri")
                xai_list = generate_xai_insights(secili_musteri, harcanan_tutar, secilen_kategori, proba)
                _kutu_sinifi = {"pos": "xai-box", "neg": "xai-box-low", "neu": "xai-box-neutral"}
                for tur, metin in xai_list:
                    st.markdown(f'<div class="{_kutu_sinifi.get(tur, "xai-box-neutral")}">{metin}</div>', unsafe_allow_html=True)

                st.divider()

                # --- D. AKSİYON ÖNERİLERİ ---
                st.markdown("#### 💡 Kasiyer İçin Anlık Aksiyon Önerileri")
                if harcanan_tutar < 500:
                    kalan = 500 - harcanan_tutar
                    st.info(f"🎁 **Kampanya Fırsatı:** Sepete **{kalan:.2f} TL** daha eklendiğinde 50 TL Kasa İndirimi kazanılıyor!")
                elif harcanan_tutar < 1000:
                    st.success("🎉 Müşteri 500 TL üzeri kargo/indirim limitine ulaştı! VIP hediye çeki sunulabilir.")

                st.warning(f"👉 **Önerilen Kasa Önü Ürünü:** {oneri_urun} — **Özel Fiyat:** {oneri_fiyat:.2f} TL")

                st.divider()

                # --- E. GERİ BİLDİRİM DÖNGÜSÜ (SQLite'a kalıcı yazılır) ---
                st.markdown("#### 🔄 Müşteri Yanıtı Kaydı (Feedback Loop)")
                st.caption("Kasiyer teklifi sunduktan sonra müşterinin kararını kaydedin. Bu veriler kalıcı olarak veritabanına yazılır ve ROI analizini besler.")

                if st.session_state.last_feedback_msg:
                    (st.success if st.session_state.last_feedback_type == "success" else st.info)(st.session_state.last_feedback_msg)

                _ad_kayit = secili_musteri.get('Ad_Soyad') if secili_musteri else "Anonim Müşteri"
                _id_kayit = secili_musteri.get('Musteri_ID') if secili_musteri else None

                fb1, fb2 = st.columns(2)
                with fb1:
                    if st.button("✅ Müşteri Öneriyi Kabul Etti", use_container_width=True, type="secondary"):
                        st.session_state.total_recommendations += 1
                        st.session_state.accepted_recommendations += 1
                        st.session_state.ai_generated_revenue += oneri_fiyat
                        oneri_kaydet(_id_kayit, _ad_kayit, secilen_kategori, oneri_urun, harcanan_tutar, proba, "Kabul")
                        if musteri_kimlik_input and secili_musteri:
                            musteri_satis_kaydet(musteri_kimlik_input, harcanan_tutar + oneri_fiyat, secilen_kategori)
                        st.session_state.last_feedback_msg = f"🎉 **Kabul Kaydedildi:** Sepete +{oneri_fiyat:.2f} TL eklendi ve veritabanına işlendi!"
                        st.session_state.last_feedback_type = "success"
                        st.rerun()
                with fb2:
                    if st.button("❌ Müşteri Öneriyi Reddetti", use_container_width=True):
                        st.session_state.total_recommendations += 1
                        st.session_state.rejected_recommendations += 1
                        oneri_kaydet(_id_kayit, _ad_kayit, secilen_kategori, oneri_urun, harcanan_tutar, proba, "Red")
                        if musteri_kimlik_input and secili_musteri:
                            musteri_satis_kaydet(musteri_kimlik_input, harcanan_tutar, secilen_kategori)
                        st.session_state.last_feedback_msg = "ℹ️ **Red Kaydedildi:** Yanıt model iyileştirme veri havuzuna aktarıldı."
                        st.session_state.last_feedback_type = "info"
                        st.rerun()
        else:
            st.info("👈 Önerileri ve XAI detaylarını görmek için lütfen sol taraftaki **'AI Önerilerini ve Tahmini Hesapla'** butonuna basın.")

# ------------------------------------------
# TAB 2: MAĞAZA & ROI ANALİTİĞİ (GİZLİLİK UYUMLU)
# ------------------------------------------
with tab_analitik:
    st.subheader("📊 Mağaza Performansı & AI İş Etkisi")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM musteriler")
    total_cust = cursor.fetchone()[0]
    cursor.execute("SELECT AVG(Sadakat_Puani) FROM musteriler")
    avg_puan = cursor.fetchone()[0] or 0.0
    cursor.execute("SELECT AVG(Sepet_Tutari_TL) FROM musteriler")
    avg_sepet = cursor.fetchone()[0] or 0.0
    conn.close()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Veritabanındaki Müşteri", total_cust)
    m2.metric("Ort. Sadakat Skor Puanı", f"{avg_puan:.1f}")
    m3.metric("Kasa Dönüşüm Oranı", f"%{acc_rate:.1f}")
    m4.metric("AI Kazanımı Ciro", f"{st.session_state.ai_generated_revenue:.2f} TL")

    st.divider()
    m5, m6 = st.columns(2)
    m5.metric("Ort. Sepet Tutarı (Veritabanı)", f"{avg_sepet:.1f} TL")
    m6.metric("VIP Eşiği (Üst %25 Sadakat Puanı)", f"{VIP_ESIK_DEGERI:.0f}")

    st.divider()
    st.markdown("##### 🕓 Son Öneriler (Canlı Kayıt)")
    son_kayitlar = son_onerileri_getir(8)
    if son_kayitlar:
        df_log = pd.DataFrame(son_kayitlar)[["zaman", "ad_soyad", "kategori", "urun", "tutar", "olasilik", "sonuc"]]
        df_log["ad_soyad"] = df_log["ad_soyad"].fillna("Anonim Müşteri").apply(lambda x: mask_text(str(x)) if maskeleme_aktif else x)
        df_log["olasilik"] = (df_log["olasilik"] * 100).round(1).astype(str) + "%"
        st.dataframe(df_log, use_container_width=True, hide_index=True)
    else:
        st.caption("Henüz kaydedilmiş bir öneri yanıtı yok.")

    st.divider()
    st.success("🔒  Müşteri kişisel verilerinin korunması amacıyla açık müşteri listesi ekranı kaldırılmıştır. Tüm aramalar kasa ekranından anlık olarak yapılmaktadır.")

# ------------------------------------------
# TAB 3: YENİ MÜŞTERİ KAYDI (SQLITE ENTEGRELİ)
# ------------------------------------------
with tab_yeni_musteri:
    st.subheader("➕ Veritabanına Yeni Müşteri Ekle")
    st.caption("Yeni kaydedilen müşteriler otomatik olarak nüfus-geneli ortalama profiliyle başlatılır ve model sütunlarıyla tam uyumlu hale getirilir.")

    with st.form("yeni_musteri_form"):
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            yeni_ad = st.text_input("Ad Soyad:", placeholder="Örn. Ayşe Yılmaz")
            yeni_tel = st.text_input("Telefon Numarası:", placeholder="Örn. 05551234567")
            yeni_segment = st.selectbox("Başlangıç Segmenti:", ["Yeni Müşteri", "Standart Müşteri", "Sadık Müşteri", "VIP Müşteri"])
            yeni_cinsiyet = st.selectbox("Cinsiyet:", ["Kadın", "Erkek", "Diğer"])
        with col_f2:
            yeni_yas = st.selectbox("Yaş Grubu:", ["18-25", "26-35", "36-50", "50+"])
            yeni_magaza = st.selectbox("Mağaza Tipi:", ["AVM", "Cadde", "Pop-up", "Online"])
            yeni_mevsim = st.selectbox("Mevsim:", ["İlkbahar", "Yaz", "Sonbahar", "Kış"])

        submit_form = st.form_submit_button("💾 Müşteriyi Veritabanına Kaydet", type="primary")

        if submit_form:
            if not yeni_ad.strip() or not yeni_tel.strip():
                st.error("❌ Lütfen Ad Soyad ve Telefon numarası alanlarını doldurun!")
            else:
                basari, mesaj, olusturulan_id = yeni_musteri_ekle(
                    yeni_ad, yeni_tel, yeni_segment, yeni_cinsiyet,
                    yeni_yas, yeni_magaza, yeni_mevsim, GENEL_MUSTERI_PROFILI
                )
                if basari:
                    st.success(f"🎉 {mesaj} (Atanan Müşteri ID: `{olusturulan_id}`)")
                else:
                    st.error(f"❌ {mesaj}")
