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
        color: #14532D !important;
    }
    .xai-box * {
        color: #14532D !important;
    }
    .xai-box-low {
        background-color: #FEF2F2;
        border-left: 4px solid #DC2626;
        padding: 12px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
        color: #7F1D1D !important;
    }
    .xai-box-low * {
        color: #7F1D1D !important;
    }
    .xai-box-neutral {
        background-color: #FFFBEB;
        border-left: 4px solid #D97706;
        padding: 12px 18px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 10px;
        color: #78350F !important;
    }
    .xai-box-neutral * {
        color: #78350F !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. MODEL ŞEMASI SABİTLERİ (smart_kasa_model.pkl ile birebir uyumlu)
# ==========================================
# 'Onerilen_Kategori' görünen adı <-> sütun öneki (örn. "Cilt Bakımı" -> "Cilt_Bakimi").
# Model 10 kategori üzerinde eğitildi; DB şeması, XAI ve ürün önerisi TEK bu kaynaktan besleniyor (tutarlılık için).
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

# DİNAMİK ÜRÜN HAVUZU (Kategori Bazlı Ürünler, Stok, Fiyat ve Kâr Marjı)
KATEGORI_URUN_HAVUZU = {
    "Makyaj": [
        {"urun": "Göz Farı Paleti", "fiyat": 79.90, "stok": True, "kar_marji": 0.40},
        {"urun": "Likit Ruj", "fiyat": 59.90, "stok": True, "kar_marji": 0.35},
        {"urun": "Hacim Veren Maskara", "fiyat": 89.90, "stok": True, "kar_marji": 0.50},
    ],
    "Cilt Bakımı": [
        {"urun": "Misel Su (200ml)", "fiyat": 69.90, "stok": True, "kar_marji": 0.45},
        {"urun": "Nemlendirici Yüz Kremi", "fiyat": 129.90, "stok": True, "kar_marji": 0.50},
        {"urun": "Güneş Koruyucu Krem", "fiyat": 149.90, "stok": True, "kar_marji": 0.40},
    ],
    "Ağız Bakımı": [
        {"urun": "Ağız Çalkalama Suyu", "fiyat": 44.90, "stok": True, "kar_marji": 0.30},
        {"urun": "Diş Beyazlatıcı Macun", "fiyat": 54.90, "stok": True, "kar_marji": 0.40},
    ],
    "Parfüm": [
        {"urun": "Vücut Spreyi (Body Mist)", "fiyat": 59.90, "stok": True, "kar_marji": 0.35},
        {"urun": "Cep Parfümü (15ml)", "fiyat": 89.90, "stok": True, "kar_marji": 0.45},
    ],
    "Saç Bakımı": [
        {"urun": "Kuru Şampuan", "fiyat": 64.90, "stok": True, "kar_marji": 0.40},
        {"urun": "Saç Bakım Yağı", "fiyat": 99.90, "stok": True, "kar_marji": 0.50},
    ],
    "Vücut Bakımı & Banyo": [
        {"urun": "Banyo Topu", "fiyat": 29.90, "stok": True, "kar_marji": 0.30},
        {"urun": "Nemlendirici Vücut Losyonu", "fiyat": 79.90, "stok": True, "kar_marji": 0.40},
    ],
    "El & Ayak Bakımı": [
        {"urun": "Aseton (Besleyici)", "fiyat": 24.90, "stok": True, "kar_marji": 0.25},
        {"urun": "Yoğun El Kremi", "fiyat": 49.90, "stok": True, "kar_marji": 0.35},
    ],
    "Güneş & Bronzlaşma": [
        {"urun": "Aloe Vera Jeli", "fiyat": 39.90, "stok": True, "kar_marji": 0.35},
        {"urun": "Bronzlaştırıcı Yağ", "fiyat": 119.90, "stok": True, "kar_marji": 0.45},
    ],
    "Aksesuar & Güzellik Aletleri": [
        {"urun": "Gua Sha Taşı", "fiyat": 89.90, "stok": True, "kar_marji": 0.50},
        {"urun": "Makyaj Süngeri Seti", "fiyat": 49.90, "stok": True, "kar_marji": 0.40},
    ],
    "Erkek Bakım": [
        {"urun": "Tıraş Köpüğü", "fiyat": 54.90, "stok": True, "kar_marji": 0.35},
        {"urun": "Tıraş Sonrası Balsam", "fiyat": 79.90, "stok": True, "kar_marji": 0.40},
    ],
}

def dinamik_urun_oner(secilen_kategori, sepet_tutari=0.0):
    """Kategori havuzundan sepet tutarına ve kârlılığa uygun ürünü dinamik olarak seçer."""
    havuz = KATEGORI_URUN_HAVUZU.get(secilen_kategori, [])
    if not havuz:
        return "Genel Fırsat Ürünü", 49.90

    # Sepet tutarına göre bütçe üst sınırı (Sepetin max %30'u veya en az 50 TL)
    butce_siniri = max(sepet_tutari * 0.30, 50.0)
    uygun_urunler = [
        u for u in havuz if u["stok"] and u["fiyat"] <= butce_siniri
    ]

    # Bütçeye uygun ürün yoksa stoktaki en ucuz ürünü seç
    if not uygun_urunler:
        uygun_urunler = sorted(
            [u for u in havuz if u["stok"]], key=lambda x: x["fiyat"]
        )

    if not uygun_urunler:
        return "Kasa Önü Fırsat Ürünü", 39.90

    # En yüksek kâr marjına sahip ürünü seç
    en_iyi_urun = max(uygun_urunler, key=lambda x: x["kar_marji"])
    return en_iyi_urun["urun"], en_iyi_urun["fiyat"]

# Kategorik özellikler. NOT: Hafta_Ici_Hafta_Sonu veride 0/1 TAM SAYI olarak
# saklanıyor ve model TAM SAYI (int) kategorileriyle eğitildi (0, 1) — metne
# çevrilirse (örn. "Hafta Sonu") OneHotEncoder bunu "bilinmeyen kategori"
# sayıp sessizce sıfırlar. Bu yüzden DB'de de INTEGER olarak tutuluyor.
MUSTERI_KATEGORIK_KOLONLAR = [
    "Cinsiyet",
    "Yas_Grubu",
    "Magaza_Tipi",
    "Mevsim",
    "Islem_Saati_Dilimi",
    "Hafta_Ici_Hafta_Sonu",
]
_KATEGORIK_SQL_TIPI = {"Hafta_Ici_Hafta_Sonu": "INTEGER"}  # diğerleri TEXT

MUSTERI_GENEL_SAYISAL_KOLONLAR = [
    "Sadakat_Puani",
    "Promosyon_Hassasiyeti_Skoru",
    "Gecen_Gun_Sayisi",
    "Coklu_Kategori_Alim_Skoru",
    "Sepet_Tutari_TL",
]

MUSTERI_KATEGORI_SAYISAL_KOLONLAR = []
for _onek in KATEGORI_KOLON_ONEKI.values():
    MUSTERI_KATEGORI_SAYISAL_KOLONLAR += [
        f"{_onek}_Alisveris_Sayisi",
        f"Gecen_Gun_{_onek}",
        f"{_onek}_Ort_Alim_Araligi",
        f"{_onek}_Tuketim_Orani",
    ]

MUSTERI_TUM_SAYISAL_KOLONLAR = (
    MUSTERI_GENEL_SAYISAL_KOLONLAR + MUSTERI_KATEGORI_SAYISAL_KOLONLAR
)
# DB'deki TÜM müşteri öznitelik sütunları (kimlik hariç) - modelin ihtiyaç
# duyduğu her şeyi kapsar; init_database() bu listeyi eksiksiz oluşturur.
MUSTERI_TUM_OZNITELIK_KOLONLARI = (
    MUSTERI_KATEGORIK_KOLONLAR + MUSTERI_TUM_SAYISAL_KOLONLAR
)

# ==========================================
# 3. VERİTABANI (SQLITE) MİMARİSİ VE YÖNETİMİ
# ==========================================
DB_FILE = "kasa_veritabani.db"
MUSTERI_CSV_YOLU = "musteri_davranis_seti.csv"


def _telefon_normallestir(deger):
    """Farklı telefon yazımlarını (05xx..., 5xx..., +90..., boşluklu/tireli)
    tek, karşılaştırılabilir bir formata indirger: sadece rakamlar, başındaki
    ülke kodu (90) veya yurt-içi sıfır (0) atılır. Böylece kullanıcı telefonu
    hangi formatta yazarsa yazsın (veya CSV'de hangi formatta saklanmışsa)
    arama tutarlı çalışır."""
    if deger is None:
        return ""
    rakamlar = "".join(ch for ch in str(deger) if ch.isdigit())
    if len(rakamlar) == 12 and rakamlar.startswith("90"):
        rakamlar = rakamlar[2:]
    if len(rakamlar) == 11 and rakamlar.startswith("0"):
        rakamlar = rakamlar[1:]
    return rakamlar


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
            if "telefon" in df_csv.columns:
                df_to_save.insert(1, "telefon", df_csv["telefon"].apply(_telefon_normallestir))
            df_to_save.to_sql("musteriler", conn, if_exists="append", index=False)
            conn.commit()
        except Exception as e:
            st.error(f"CSV'den SQLite'a veri aktarılırken hata: {e}")

    conn.close()
    _senkronize_telefon_csvden()


def _senkronize_telefon_csvden():
    """CSV'de olup (örn. sonradan eklenmiş) veritabanındaki müşteri
    kayıtlarında hâlâ eksik olan telefon numaralarını tamamlar. Sadece DB'de
    telefon NULL/boş olan satırlar güncellenir; mevcut değerlerin (örn.
    uygulama içinden kayıt olmuş yeni müşteriler) üzerine asla yazılmaz.
    DB zaten güncelse (eksik yoksa) hemen çıkar, her yenilemede performans
    kaybı yaratmaz."""
    if not os.path.exists(MUSTERI_CSV_YOLU):
        return
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("PRAGMA table_info(musteriler)")
        if "telefon" not in {r[1] for r in cursor.fetchall()}:
            return

        cursor.execute("SELECT COUNT(*) FROM musteriler WHERE telefon IS NULL OR telefon = ''")
        if cursor.fetchone()[0] == 0:
            return  # zaten senkron, hızlı çıkış

        df_csv = pd.read_csv(MUSTERI_CSV_YOLU, encoding="utf-8-sig", dtype={"Musteri_ID": str})
        df_csv.columns = [str(c).strip() for c in df_csv.columns]
        if "telefon" not in df_csv.columns:
            return

        df_tel = df_csv[["Musteri_ID", "telefon"]].copy()
        df_tel["telefon"] = df_tel["telefon"].apply(_telefon_normallestir)
        df_tel = df_tel[df_tel["telefon"] != ""]

        cursor.executemany(
            "UPDATE musteriler SET telefon = ? WHERE Musteri_ID = ? AND (telefon IS NULL OR telefon = '')",
            list(zip(df_tel["telefon"], df_tel["Musteri_ID"]))
        )
        conn.commit()
    except Exception as e:
        st.warning(f"Telefon numaraları senkronize edilirken uyarı: {e}")
    finally:
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
    Telefon için normalize edilmiş (sadece rakam, başında 0/90 olmadan) hâli
    karşılaştırılır; böylece '05551234567', '5551234567' ve '+905551234567'
    hepsi aynı müşteriyi bulur. Her iki değer de '?' yer tutucusu ile
    bağlanır; ham girdi asla SQL metnine dahil edilmez."""
    if not arama_terimi:
        return None
    terim_ham = arama_terimi.strip()
    terim_telefon = _telefon_normallestir(terim_ham)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM musteriler WHERE telefon = ? OR Musteri_ID = ? LIMIT 1",
        (terim_telefon, terim_ham)
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def yeni_musteri_ekle(ad_soyad, telefon, segment, cinsiyet, yas_grubu, magaza_tipi, mevsim):
    """Güvenli (parameterized) müşteri ekleme.

    TASARIM KARARI: Yeni müşteri GERÇEKTEN boş bir geçmişle başlar; nüfus
    ortalamasıyla DEĞİL. Kategori bazlı 'kaç gün önce aldı', 'ortalama
    alım aralığı', 'tüketim oranı' alanları bu müşteri için henüz
    TANIMSIZ olduğundan NULL bırakılır — nüfus ortalamasını bu kişinin
    KENDİ geçmişiymiş gibi göstermek yanıltıcı olurdu. Alışveriş sayısı
    (Alisveris_Sayisi) ve kategori-çeşitliliği skoru gerçekten 0'dır, bu
    yüzden 0 olarak yazılır (0, NULL'dan farklı olarak burada DOĞRU bir
    bilgidir). Müşteri gerçekten kasada işlem yaptıkça (bkz.
    musteri_satis_kaydet), bu alanlar GERÇEK verilerle dolmaya başlar.
    Formda toplanmayan diğer alanlar (Promosyon_Hassasiyeti_Skoru,
    Islem_Saati_Dilimi vb.) de bilinmediği için NULL bırakılır; tahmin
    anında bunlar pipeline'ın EĞİTİMDEN öğrendiği medyan/mod değerleriyle
    doldurulur (SimpleImputer) — bu, nüfus ortalamasını sahte kişisel
    geçmiş gibi sunmaktan daha dürüst bir yaklaşımdır.
    """
    puan_map = {"Yeni Müşteri": 10.0, "Standart Müşteri": 30.0, "Sadık Müşteri": 65.0, "VIP Müşteri": 90.0}
    puan = puan_map.get(segment, 20.0)
    yeni_id = f"MST-N{random.randint(10000, 99999)}"

    # Kayıt sözlüğü: anahtarlar HER ZAMAN kod-içi sabit listelerden gelir,
    # kullanıcı girdisinden asla sütun adı türetilmez. Sadece DEĞERLER
    # (ad_soyad, telefon gibi) kullanıcıdan gelir ve '?' ile parametrize edilir.
    kayit = {
        "Cinsiyet": cinsiyet, "Yas_Grubu": yas_grubu,
        "Magaza_Tipi": magaza_tipi, "Mevsim": mevsim,
        "Sadakat_Puani": puan, "Sepet_Tutari_TL": 0.0,
        "Coklu_Kategori_Alim_Skoru": 0.0,  # gerçekten 0: henüz hiçbir kategoriden alışverişi yok
        # Aşağıdakiler kasıtlı olarak None/NULL: bu müşterinin henüz GERÇEK
        # bir davranış geçmişi yok, bu yüzden özel bir sayı uydurmuyoruz.
        "Gecen_Gun_Sayisi": None,
        "Promosyon_Hassasiyeti_Skoru": None,
        "Islem_Saati_Dilimi": None,
        "Hafta_Ici_Hafta_Sonu": None,
    }
    for onek in KATEGORI_KOLON_ONEKI.values():
        kayit[f"{onek}_Alisveris_Sayisi"] = 0       # gerçek: 0 kez alışveriş
        kayit[f"Gecen_Gun_{onek}"] = None            # tanımsız: hiç alımı yok
        kayit[f"{onek}_Ort_Alim_Araligi"] = None      # tanımsız: aralık hesaplanamaz
        kayit[f"{onek}_Tuketim_Orani"] = None         # tanımsız: oran hesaplanamaz

    kolonlar = ["Musteri_ID", "telefon", "Ad_Soyad"] + [k for k in MUSTERI_TUM_OZNITELIK_KOLONLARI if k in kayit]
    degerler = [yeni_id, _telefon_normallestir(telefon), ad_soyad.strip()] + [kayit[k] for k in kolonlar[3:]]
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
    kimlik_ham = musteri_kimlik.strip()
    kimlik_telefon = _telefon_normallestir(kimlik_ham)
    onek = KATEGORI_KOLON_ONEKI.get(secilen_kategori)  # sabit, kod-içi eşlemeden -> güvenli

    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        if onek:
            kolon_gecen_gun = f"Gecen_Gun_{onek}"
            kolon_alisveris = f"{onek}_Alisveris_Sayisi"
            kolon_oran = f"{onek}_Tuketim_Orani"
            # Ort_Alim_Araligi'ye kasıtlı dokunulmuyor: gerçek bir aralık ancak
            # bu kategoride EN AZ 2. alımda hesaplanabilir (önceki alım
            # tarihini bilmemiz gerekir). İlk alımda NULL kalması, uydurma bir
            # sayı yazmaktan daha doğrudur; 2. alımdan itibaren zaten dolu olan
            # değer üzerinden tüketim oranı hesaplanmaya devam eder.
            cursor.execute(f"""
                UPDATE musteriler
                SET Sepet_Tutari_TL = ?,
                    Gecen_Gun_Sayisi = 0,
                    {kolon_gecen_gun} = 0,
                    {kolon_oran} = 0,
                    {kolon_alisveris} = COALESCE({kolon_alisveris}, 0) + 1
                WHERE telefon = ? OR Musteri_ID = ?
            """, (harcanan_tutar, kimlik_telefon, kimlik_ham))
        else:
            cursor.execute("""
                UPDATE musteriler SET Sepet_Tutari_TL = ?, Gecen_Gun_Sayisi = 0
                WHERE telefon = ? OR Musteri_ID = ?
            """, (harcanan_tutar, kimlik_telefon, kimlik_ham))
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
    """Zengin, çok sinyalli açıklanabilirlik kutuları üretir."""
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
        alisveris_ham = secili_musteri.get(f"{kat_onek}_Alisveris_Sayisi")
        alisveris_sayisi = float(alisveris_ham) if alisveris_ham not in (None, "") else 0.0

        if alisveris_sayisi == 0:
            # Bu kategoride GERÇEKTEN hiç alışverişi yok; "X gün önce aldı"
            # gibi bir iddiada bulunmuyoruz çünkü tanımsız (uydurma olurdu).
            insights.append(("neu", f"🆕 **İlk Kez ({secilen_kategori}):** Müşterinin bu kategoride geçmiş alışverişi yok; öneri keşif amaçlı olabilir."))
        else:
            gecen_gun_ham = secili_musteri.get(f"Gecen_Gun_{kat_onek}")
            ort_aralik_ham = secili_musteri.get(f"{kat_onek}_Ort_Alim_Araligi")
            tuketim_orani_ham = secili_musteri.get(f"{kat_onek}_Tuketim_Orani")

            if gecen_gun_ham not in (None, "") and ort_aralik_ham not in (None, "", 0):
                gecen_gun = float(gecen_gun_ham)
                ort_aralik = float(ort_aralik_ham)
                if gecen_gun >= ort_aralik:
                    insights.append(("pos", f"⏳ **Yenileme Zamanı Gelmiş ({secilen_kategori}):** Son alımdan bu yana {int(gecen_gun)} gün geçmiş (Ort. Döngü: {int(ort_aralik)} gün). Müşterinin bu ürüne ihtiyacı yüksek."))
                else:
                    insights.append(("neu", f"🕐 **Henüz Erken ({secilen_kategori}):** Son alımdan {int(gecen_gun)} gün geçmiş, ortalama döngüsü {int(ort_aralik)} gün — henüz vaktinden önce olabilir."))

            if tuketim_orani_ham not in (None, "") and float(tuketim_orani_ham) > 0.6:
                insights.append(("pos", f"📈 **Yüksek Tüketim Skoru:** {secilen_kategori} kategorisindeki geçmiş tüketim oranı yüksek."))

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
    st.caption("")
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
                oneri_urun, oneri_fiyat = dinamik_urun_oner(secilen_kategori, harcanan_tutar)

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
                # Yeni/kısmi müşteri kayıtlarında bazı alanlar SQL NULL (Python None)
                # olabilir (örn. hiç alışverişi olmayan bir kategori). Bunları np.nan'a
                # çeviriyoruz ki pipeline'ın SimpleImputer'ı bunları tanıyıp EĞİTİMDEN
                # öğrendiği medyan/mod değerleriyle doldursun (sıfır veya uydurma bir
                # sayı ATAMIYORUZ, gerçek eksik-veri doldurma mekanizmasını kullanıyoruz).
                df_input = df_input.replace({None: np.nan})
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
    m1.metric("Kayıtlı Müşteri", total_cust)
    m2.metric("Ort. Sadakat Skor Puanı", f"{avg_puan:.1f}")
    m3.metric("Kasa Dönüşüm Oranı", f"%{acc_rate:.1f}")
    m4.metric("AI Kazanımı Ciro", f"{st.session_state.ai_generated_revenue:.2f} TL")

    st.divider()
    m5, m6 = st.columns(2)
    m5.metric("Ort. Sepet Tutarı", f"{avg_sepet:.1f} TL")
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
    st.subheader("➕ Yeni Müşteri Kayıt Formu")
    st.caption("Buradan eklenen müşteriler anında kasa ekranından telefon/ID ile sorgulanabilir.")

    with st.form("yeni_musteri_formu", clear_on_submit=True):
        f_ad = st.text_input("Ad Soyad:")
        f_tel = st.text_input("Telefon No (Örn: 05551234567):")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            f_cinsiyet = st.selectbox("Cinsiyet:", ["Kadın", "Erkek"])
            f_magaza = st.selectbox("Mağaza Tipi:", ["AVM", "Cadde"])
        with col_f2:
            f_yas = st.selectbox("Yaş Grubu:", ["18-24", "25-34", "35-49", "50+"])
            f_mevsim = st.selectbox("Kayıt Mevsimi:", ["Yaz", "Kış", "İlkbahar", "Sonbahar"])
        f_segment = st.selectbox("Müşteri Segmenti:", ["Yeni Müşteri", "Standart Müşteri", "Sadık Müşteri", "VIP Müşteri"])

        submit_btn = st.form_submit_button("💾 Müşteriyi Kaydet")

        if submit_btn:
            if f_ad and f_tel:
                basari, mesaj, yeni_id = yeni_musteri_ekle(f_ad, f_tel, f_segment, f_cinsiyet, f_yas, f_magaza, f_mevsim)
                if basari:
                    st.success(f"✅ {mesaj} (Müşteri: {f_ad} — ID: {yeni_id})")
                    st.cache_data.clear()
                else:
                    st.error(f"⚠️ {mesaj}")
            else:
                st.warning("Lütfen Ad Soyad ve Telefon alanlarını boş bırakmayın.")
