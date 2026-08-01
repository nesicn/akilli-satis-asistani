import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sqlite3
import os

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
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. VERİTABANI (SQLITE) MİMARİSİ VE YÖNETİMİ
# ==========================================
DB_FILE = "kasa_veritabani.db"
MUSTERI_CSV_YOLU = "musteri_davranis_seti.csv"

def get_db_connection():
    """SQLite veritabanı bağlantısı oluşturur."""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Veritabanını başlatır. Yoksa oluşturur ve CSV'deki verileri aktarır."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Müşteriler tablosunu oluştur
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS musteriler (
            Musteri_ID TEXT PRIMARY KEY,
            telefon TEXT UNIQUE,
            Ad_Soyad TEXT,
            Cinsiyet TEXT,
            Yas_Grubu TEXT,
            Magaza_Tipi TEXT,
            Mevsim TEXT,
            Hafta_Ici_Hafta_Sonu TEXT,
            Islem_Saati_Dilimi TEXT,
            Sadakat_Puani REAL,
            Sepet_Tutari_TL REAL,
            Gecen_Gun_Sayisi REAL
        )
    """)
    conn.commit()

    # Tablo boşsa CSV'den verileri aktar
    cursor.execute("SELECT COUNT(*) FROM musteriler")
    count = cursor.fetchone()[0]
    
    if count == 0 and os.path.exists(MUSTERI_CSV_YOLU):
        try:
            df_csv = pd.read_csv(MUSTERI_CSV_YOLU, encoding="utf-8-sig", dtype={"Musteri_ID": str, "telefon": str})
            df_csv.columns = [str(c).strip() for c in df_csv.columns]
            
            # Veritabanında olan sütunları seçip kaydet
            valid_cols = [c for c in df_csv.columns if c in [
                "Musteri_ID", "telefon", "Ad_Soyad", "Cinsiyet", "Yas_Grubu", 
                "Magaza_Tipi", "Mevsim", "Hafta_Ici_Hafta_Sonu", "Islem_Saati_Dilimi", 
                "Sadakat_Puani", "Sepet_Tutari_TL", "Gecen_Gun_Sayisi"
            ]]
            df_to_save = df_csv[valid_cols]
            df_to_save.to_sql("musteriler", conn, if_exists="append", index=False)
            conn.commit()
        except Exception as e:
            st.error(f"CSV'den SQLite'a veri aktarırken hata: {e}")
            
    conn.close()

# Veritabanını ilkle
init_database()

def musteri_ara_telefon(telefon_no):
    """Güvenli (Parameterized SQL) Müşteri Arama."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM musteriler WHERE telefon = ?", (telefon_no.strip(),))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None

def yeni_musteri_ekle(ad_soyad, telefon, segment):
    """Güvenli Müşteri Ekleme Fonksiyonu."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    puan_map = {"Yeni Müşteri": 10.0, "Standart Müşteri": 30.0, "Sadık Müşteri": 65.0, "VIP Müşteri": 90.0}
    puan = puan_map.get(segment, 20.0)
    
    import random
    yeni_id = f"M{random.randint(10000, 99999)}"
    
    try:
        cursor.execute("""
            INSERT INTO musteriler (Musteri_ID, telefon, Ad_Soyad, Sadakat_Puani, Sepet_Tutari_TL, Gecen_Gun_Sayisi)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (yeni_id, telefon.strip(), ad_soyad.strip(), puan, 0.0, 0))
        conn.commit()
        conn.close()
        return True, "Müşteri veritabanına başarıyla eklendi!"
    except sqlite3.IntegrityError:
        conn.close()
        return False, "Bu telefon numarasıyla kayıtlı bir müşteri zaten var!"
    except Exception as e:
        conn.close()
        return False, f"Hata oluştu: {str(e)}"

def musteri_satis_kaydet(telefon, harcanan_tutar):
    """Müşteri alışveriş yaptığında veritabanındaki verilerini canlı günceller."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            UPDATE musteriler 
            SET Sepet_Tutari_TL = ?,
                Gecen_Gun_Sayisi = 0
            WHERE telefon = ?
        """, (harcanan_tutar, telefon.strip()))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        conn.close()
        return False

# ==========================================
# 3. SESSION STATE İLKLEME
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

def reset_analysis():
    st.session_state.analiz_yapildi = False

# ==========================================
# 4. MODEL YÜKLEME
# ==========================================
MODEL_YOLU = "smart_kasa_model.pkl"

@st.cache_resource
def load_model(model_path=MODEL_YOLU):
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except:
        return None

model = load_model()

# Sabitler
KATEGORI_URUN_ONERI = {
    "Makyaj": ("Göz Farı", 79.90),
    "Cilt Bakımı": ("Misel Su", 69.90),
    "Ağız Bakımı": ("Ağız Çalkalama Suyu", 44.90),
    "Parfüm": ("Vücut Spreyi", 59.90),
    "Saç Bakımı": ("Kuru Şampuan", 64.90),
}

# ==========================================
# 5. YAN MENÜ (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/checkout.png", width=70)
    st.title("Smart Checkout AI")
    st.caption("SQLite & AI Powered POS")
    st.divider()

    st.markdown("**Sistem Durumu:**")
    if model is not None:
        st.success("🟢 AI Modeli & SQLite Aktif")
    else:
        st.warning("🟠 Model Bulunamadı (Simülasyon Modu)")

    st.divider()
    st.markdown("**📈 Canlı ROI Metrikleri:**")
    st.metric("Ek Ciro Katkısı", f"{st.session_state.ai_generated_revenue:.2f} TL")
    acc_rate = (st.session_state.accepted_recommendations / st.session_state.total_recommendations * 100) if st.session_state.total_recommendations > 0 else 0.0
    st.metric("Kasa Dönüşüm Oranı", f"%{acc_rate:.1f}")

# ==========================================
# 6. ANA EKRAN VE SEKMELER
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

        musteri_tel_input = st.text_input(
            "📱 Müşteri Telefon No:",
            placeholder="05xxxxxxxxx",
            key="input_musteri_tel",
            on_change=reset_analysis
        )

        secili_musteri = None
        if musteri_tel_input:
            secili_musteri = musteri_ara_telefon(musteri_tel_input)
            if secili_musteri:
                st.success(f"👤 Müşteri Bulundu: **{secili_musteri.get('Ad_Soyad')}**")
            else:
                st.info("ℹ️ Kayıtlı müşteri bulunamadı. Genel profil ile işlem yapılıyor.")

        st.divider()

        harcanan_tutar = st.number_input("💰 Sepet Tutarı (TL):", min_value=10.0, value=350.0, step=10.0, on_change=reset_analysis)
        secilen_kategori = st.selectbox("🏷️ Ağırlıklı Kategori:", list(KATEGORI_URUN_ONERI.keys()), on_change=reset_analysis)

        if st.button("⚡ AI Önerilerini Hesapla", type="primary", use_container_width=True):
            st.session_state.analiz_yapildi = True

    with col_right:
        st.subheader("2. Yapay Zeka Karar Destek")

        if st.session_state.analiz_yapildi:
            oneri_urun, oneri_fiyat = KATEGORI_URUN_ONERI.get(secilen_kategori, ("Kasa Önü Fırsat Ürünü", 29.90))
            
            # Basitleştirilmiş Tahmin / Skor Mantığı
            proba = 0.78 if (secili_musteri and secili_musteri.get('Sadakat_Puani', 0) > 50) else 0.52

            st.markdown("#### 🎯 İkna Olasılığı Skoru")
            st.progress(proba)
            st.caption(f"Tahmin Edilen İkna Oranı: **%{proba*100:.0f}**")

            st.divider()
            st.markdown("#### 💡 Kasiyer İçin Öneri")
            st.warning(f"👉 **Tavsiye Edilen Ürün:** {oneri_urun} — **Fiyat:** {oneri_fiyat:.2f} TL")

            st.divider()
            st.markdown("#### 🔄 Müşteri Yanıtı")
            fb_c1, fb_c2 = st.columns(2)
            
            with fb_c1:
                if st.button("✅ Öneriyi Kabul Etti", use_container_width=True):
                    toplam_tutar = harcanan_tutar + oneri_fiyat
                    
                    # 1. Session State Metriklerini Güncelle
                    st.session_state.total_recommendations += 1
                    st.session_state.accepted_recommendations += 1
                    st.session_state.ai_generated_revenue += oneri_fiyat
                    
                    # 2. Kayıtlı Müşteri İse SQLite Veritabanını Güncelle
                    if secili_musteri and musteri_tel_input:
                        musteri_satis_kaydet(musteri_tel_input, toplam_tutar)
                        st.success(f"✅ Satış tamamlandı (+{oneri_fiyat:.2f} TL) & Müşteri profili güncellendi!")
                    else:
                        st.success(f"✅ Satış tamamlandı (+{oneri_fiyat:.2f} TL)!")

            with fb_c2:
                if st.button("❌ Reddetti", use_container_width=True):
                    # 1. Session State Metriklerini Güncelle
                    st.session_state.total_recommendations += 1
                    st.session_state.rejected_recommendations += 1
                    
                    # 2. Öneri reddedilse bile ana sepet tutarı ile veritabanını güncelle
                    if secili_musteri and musteri_tel_input:
                        musteri_satis_kaydet(musteri_tel_input, harcanan_tutar)
                    
                    st.info("Satış ana sepet tutarı ile tamamlandı.")
        else:
            st.info("👈 Önerileri görmek için butona basın.")

# ------------------------------------------
# TAB 2: MAĞAZA & ROI ANALİTİĞİ (GİZLİLİK UYUMLU)
# ------------------------------------------
with tab_analitik:
    st.subheader("📊 Mağaza Performansı & AI İş Etkisi")
    
    # SQLite'dan genel istatistikleri çek
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM musteriler")
    total_cust = cursor.fetchone()[0]
    
    cursor.execute("SELECT AVG(Sadakat_Puani) FROM musteriler")
    avg_puan = cursor.fetchone()[0] or 0.0
    conn.close()

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Veritabanındaki Müşteri", total_cust)
    m2.metric("Ort. Sadakat Skor Puanı", f"{avg_puan:.1f}")
    m3.metric("Kasa Dönüşüm Oranı", f"%{acc_rate:.1f}")
    m4.metric("AI Kazanımı Ciro", f"{st.session_state.ai_generated_revenue:.2f} TL")

    st.divider()
    st.success("🔒 **Privacy by Design:** Müşteri kişisel verilerinin korunması (KVKK) amacıyla açık müşteri listesi ekran kaldırılmıştır. Tüm aramalar kasa ekranından anlık olarak yapılmaktadır.")

# ------------------------------------------
# TAB 3: YENİ MÜŞTERİ KAYDI (SQLITE ENTEGRELİ)
# ------------------------------------------
with tab_yeni_musteri:
    st.subheader("➕ Yeni Müşteri Kayıt Formu")
    st.caption("Buradan eklenen müşteriler anında SQLite veritabanına yazılır ve kasa ekranında sorgulanabilir.")

    with st.form("yeni_musteri_formu", clear_on_submit=True):
        f_ad = st.text_input("Ad Soyad:")
        f_tel = st.text_input("Telefon No (Örn: 05551234567):")
        f_segment = st.selectbox("Müşteri Segmenti:", ["Yeni Müşteri", "Standart Müşteri", "Sadık Müşteri", "VIP Müşteri"])

        submit_btn = st.form_submit_button("💾 Müşteriyi Veritabanına Kaydet")
        
        if submit_btn:
            if f_ad and f_tel:
                basari, mesaj = yeni_musteri_ekle(f_ad, f_tel, f_segment)
                if basari:
                    st.success(f"✅ {mesaj} (Müşteri: {f_ad})")
                else:
                    st.error(f"⚠️ {mesaj}")
            else:
                st.warning("Lütfen Ad Soyad ve Telefon alanlarını boş bırakmayın.")
