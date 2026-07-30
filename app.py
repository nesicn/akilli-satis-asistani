import streamlit as st
import pandas as pd
import numpy as np
import joblib
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

# Custom CSS Stilleri
st.markdown("""
    <style>
    .main-header {
        font-size: 26px;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
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
# 2. SESSION STATE İLKLEME & CALLBACKS
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

# Girdiler değiştiğinde analizi ve bildirimleri sıfırlayan callback
def reset_analysis():
    st.session_state.analiz_yapildi = False
    st.session_state.last_feedback_msg = None
    st.session_state.last_feedback_type = None

# ==========================================
# 3. VERİ VE MODEL YÜKLEME (CACHED)
# ==========================================
# GÜNCELLEME NOTU: Proje "akıllı" (2. nesil) veri setine taşındı.
#   Eski model : kasa_model.pkl          + musteriler.csv      (6 kategori)
#   Yeni model : smart_kasa_model.pkl    + musteri_davranis_seti.csv (10 kategori, 53 özellik)
# Arayüz (sekmeler, kartlar, XAI kutuları, feedback akışı) BİREBİR korunmuştur;
# sadece arka plandaki veri/model bağlantısı yeni şemaya göre düzeltilmiştir.
MODEL_YOLU = "smart_kasa_model.pkl"
MUSTERI_CSV_YOLU = "musteri_davranis_seti.csv"


@st.cache_resource
def load_model(model_path=MODEL_YOLU):
    """Yapay zeka modelini güvenli bir şekilde yükler."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model dosyası bulunamadı: `{model_path}`. Lütfen dosyanın proje dizininde olduğundan emin olun.")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata oluştu: {str(e)}")
        return None


@st.cache_data
def load_customers(csv_path=MUSTERI_CSV_YOLU):
    """Müşteri veri setini yükler.

    encoding='utf-8-sig': dosyanın başında BOM (byte order mark) olsa bile
    ilk sütun adı ('Musteri_ID') bozulmadan okunur; BOM yoksa da zararsızdır.
    """
    if not os.path.exists(csv_path):
        st.warning(f"⚠️ `{csv_path}` bulunamadı.")
        return pd.DataFrame(columns=["Musteri_ID", "Ad_Soyad"])
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype={"Musteri_ID": str})
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"❌ Müşteri verisi okunurken hata oluştu: {str(e)}")
        return pd.DataFrame()


model = load_model()
df_musteriler = load_customers()

# ==========================================
# 3.1 MODEL ŞEMASI SABİTLERİ (smart_kasa_model.pkl ile birebir uyumlu)
# ==========================================
# 'Onerilen_Kategori' görünen adı <-> sütun öneki (örn. "Cilt Bakımı" -> "Cilt_Bakimi").
# Modelin eğitildiği 10 kategori burada tanımlı; XAI açıklamaları ve kasa-önü
# ürün önerisi TEK bu kaynaktan besleniyor (tutarlılık için).
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

# Her kategori için kasa önünde önerilecek ürün ve fiyatı. Ekranda gösterilen
# ürünle modele giden 'Onerilen_Urun' özelliği TEK bir kaynaktan geldiği için
# her zaman tutarlıdır (eğitim verisindeki en sık görülen ürün seçildi).
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

_MODEL_SAYISAL_KOLONLAR = [
    "Sadakat_Puani", "Promosyon_Hassasiyeti_Skoru", "Gecen_Gun_Sayisi",
    "Coklu_Kategori_Alim_Skoru", "Sepet_Tutari_TL",
]
for _onek in KATEGORI_KOLON_ONEKI.values():
    _MODEL_SAYISAL_KOLONLAR += [
        f"{_onek}_Alisveris_Sayisi", f"Gecen_Gun_{_onek}",
        f"{_onek}_Ort_Alim_Araligi", f"{_onek}_Tuketim_Orani",
    ]
_MODEL_KATEGORIK_KOLONLAR = [
    "Cinsiyet", "Yas_Grubu", "Magaza_Tipi", "Mevsim",
    "Islem_Saati_Dilimi", "Hafta_Ici_Hafta_Sonu",
]


def build_default_profile(df):
    """Kayıtlı/eşleşen bir müşteri olmadığında modele SIFIRLAR yerine bu
    genel (medyan/en sık değer) profili veriyoruz; aksi halde tahmin sessizce
    anlamsızlaşır (özelliklerin büyük kısmı sıfırlanmış olurdu)."""
    profil = {}
    if df is None or df.empty:
        return profil
    for col in _MODEL_SAYISAL_KOLONLAR:
        if col in df.columns:
            deger = df[col].median()
            profil[col] = deger if pd.notna(deger) else 0
    for col in _MODEL_KATEGORIK_KOLONLAR:
        if col in df.columns and not df[col].mode().empty:
            profil[col] = df[col].mode().iloc[0]
    return profil


GENEL_MUSTERI_PROFILI = build_default_profile(df_musteriler)

# VIP / Sadık Müşteri ayrımı: bu veri setinde ayrı bir 'Segment' sütunu yok,
# bu yüzden gerçekten var olan 'Sadakat_Puani' üzerinden (üst %25 / medyan
# üstü) türetiyoruz. Eşikler veriye göre otomatik hesaplanır.
if not df_musteriler.empty and "Sadakat_Puani" in df_musteriler.columns:
    VIP_ESIK_DEGERI = df_musteriler["Sadakat_Puani"].quantile(0.75)
    SADIK_ESIK_DEGERI = df_musteriler["Sadakat_Puani"].median()
else:
    VIP_ESIK_DEGERI = 0
    SADIK_ESIK_DEGERI = 0

# "Sık alışveriş yapan müşteri" XAI içgörüsü için eşik: tüm kategorilerdeki
# alışveriş sayıları toplamının üst %25'i (veriye göre otomatik kalibre edilir).
if not df_musteriler.empty:
    _alisveris_kolonlari = [f"{onek}_Alisveris_Sayisi" for onek in KATEGORI_KOLON_ONEKI.values() if f"{onek}_Alisveris_Sayisi" in df_musteriler.columns]
    SIK_ALISVERIS_ESIGI = df_musteriler[_alisveris_kolonlari].sum(axis=1).quantile(0.75) if _alisveris_kolonlari else 15
else:
    SIK_ALISVERIS_ESIGI = 15


def musteri_segment_belirle(sadakat_puani):
    """Sadakat_Puani'na göre VIP / Sadık Müşteri / Standart etiketi üretir."""
    try:
        puan = float(sadakat_puani)
    except (TypeError, ValueError):
        return "Standart"
    if puan >= VIP_ESIK_DEGERI:
        return "VIP"
    if puan >= SADIK_ESIK_DEGERI:
        return "Sadık Müşteri"
    return "Standart"


# Maskeleme Fonksiyonu (KVKK Uyumu)
def mask_text(text, visible_chars=2):
    if not isinstance(text, str) or len(text) <= visible_chars:
        return text
    return text[:visible_chars] + "*" * (len(text) - visible_chars)


# XAI (Explainable AI) Karar Gerekçesi Oluşturucu
def generate_xai_insights(secili_musteri, harcanan_tutar, secilen_kategori, proba):
    insights = []

    sadakat_puani = secili_musteri.get('Sadakat_Puani') if secili_musteri else None
    segment = musteri_segment_belirle(sadakat_puani) if sadakat_puani is not None else 'Standart'
    if segment == 'VIP':
        insights.append("⭐ **VIP Müşteri Sadakati:** Müşterinin yüksek alışveriş bağlılığı ikna olasılığını artırıyor.")
    elif segment == 'Sadık Müşteri':
        insights.append("💙 **Sadık Müşteri Profili:** Düzenli mağaza ziyaretleri öneri kabul esnekliğini destekliyor.")

    # Gerçek sütun adları "{Kategori}_Ort_Alim_Araligi" / "{Kategori}_Tuketim_Orani"
    # (SONEK) ve "Gecen_Gun_{Kategori}" (ÖNEK) şeklindedir. KATEGORI_KOLON_ONEKI
    # üzerinden doğru case/sırada üretiyoruz; aksi halde hiçbir zaman eşleşmez.
    kat_onek = KATEGORI_KOLON_ONEKI.get(secilen_kategori)
    if secili_musteri and kat_onek:
        gecen_gun = float(secili_musteri.get(f"Gecen_Gun_{kat_onek}", 30) or 30)
        ort_aralik = float(secili_musteri.get(f"{kat_onek}_Ort_Alim_Araligi", 30) or 30)
        tuketim_orani = float(secili_musteri.get(f"{kat_onek}_Tuketim_Orani", 0.5) or 0.5)

        if ort_aralik > 0 and gecen_gun >= ort_aralik:
            insights.append(f"⏳ **Yenileme Zamanı Gelmiş ({secilen_kategori}):** Son alımdan bu yana {int(gecen_gun)} gün geçmiş (Ort. Döngü: {int(ort_aralik)} gün). Müşterinin bu ürüne ihtiyacı yüksek.")
        if tuketim_orani > 0.6:
            insights.append(f"📈 **Yüksek Tüketim Skoru:** Müşterinin {secilen_kategori} kategorisindeki geçmiş tüketim oranı (%{tuketim_orani*100:.0f}) yüksek.")

    if harcanan_tutar >= 500:
        insights.append(f"💰 **Yüksek Alışveriş Hacmi:** {harcanan_tutar:.0f} TL tutarındaki sepet, müşterinin ek tekliflere açık olduğunu gösteriyor.")
    elif harcanan_tutar < 200:
        insights.append("⚠️ **Düşük Sepet Tutarı:** Müşteri hassas bir bütçeyle alışveriş yapıyor olabilir.")

    if secili_musteri:
        # Toplam alışveriş sayısı artık tüm kategorilerdeki "_Alisveris_Sayisi"
        # sütunlarının toplamından türetiliyor (eski 'gecmis_islem_sayisi' sütunu
        # bu veri setinde yok, ama karşılığı budur).
        toplam_alisveris = sum(
            float(secili_musteri.get(f"{onek}_Alisveris_Sayisi", 0) or 0)
            for onek in KATEGORI_KOLON_ONEKI.values()
        )
        if toplam_alisveris > SIK_ALISVERIS_ESIGI:
            insights.append(f"🔄 **Sık Alışveriş Yapan Müşteri:** Geçmişteki {int(toplam_alisveris)} kategori bazlı alışveriş güven indeksini yükseltiyor.")

    if not insights:
        insights.append("ℹ️ Genel sepet ortalamaları ve mağaza içi standart müşteri profil davranışları esas alındı.")

    return insights


# ==========================================
# 4. YAN MENÜ (SIDEBAR) & CANLI METRİKLER
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/checkout.png", width=70)
    st.title("Smart Checkout AI")
    st.caption("Retail AI Assistant")
    st.divider()

    maskeleme_aktif = st.toggle("🔒 Maskeleme Modu", value=False)
    st.info("Sistem, kasa anında akıllı çapraz satış ve müşteri ikna olasılığı tahmini üretir.")

    st.divider()
    st.markdown("**Sistem Durumu:**")
    if model is not None:
        _model_nesnesi = model.named_steps.get("model", model) if hasattr(model, "named_steps") else model
        _model_adi = type(_model_nesnesi).__name__.replace("Classifier", "")
        st.success(f"🟢 AI Modeli Aktif (`{_model_adi}`)")
    else:
        st.error("🔴 AI Modeli Pasif")

    st.divider()
    st.markdown("**📈 Canlı Kasa Performansı (ROI):**")
    st.metric("Ek Ciro Katkısı", f"{st.session_state.ai_generated_revenue:.2f} TL")
    acc_rate = (st.session_state.accepted_recommendations / st.session_state.total_recommendations * 100) if st.session_state.total_recommendations > 0 else 0.0
    st.metric("Kasa Dönüşüm Oranı", f"%{acc_rate:.1f}")

# ==========================================
# 5. ANA EKRAN VE SEKMELER
# ==========================================
st.title("🛒 Akıllı Kasa Asistanı & Öneri Motoru")

tab_kasa, tab_analitik, tab_yeni_musteri = st.tabs([
    "🛍️ Kasa İşlem Ekranı",
    "📊 Müşteri & Mağaza Analitiği",
    "➕ Yeni Müşteri Kaydı"
])

# ------------------------------------------
# TAB 1: KASA İŞLEM EKRANI
# ------------------------------------------
with tab_kasa:
    col_left, col_right = st.columns([1, 1.2], gap="large")

    with col_left:
        st.subheader("1. Müşteri & Sepet Bilgileri")

        # NOT: Bu veri setinde telefon numarası yok; müşteri araması artık
        # gerçekten var olan ve benzersiz olan Musteri_ID üzerinden yapılıyor.
        musteri_id_input = st.text_input(
            "🆔 Müşteri ID (Örn: MST-00001):",
            value="MST-00001",
            key="input_musteri_id",
            on_change=reset_analysis
        )

        secili_musteri = None
        if musteri_id_input and not df_musteriler.empty and "Musteri_ID" in df_musteriler.columns:
            match = df_musteriler[df_musteriler["Musteri_ID"].astype(str).str.contains(musteri_id_input.strip(), case=False, na=False)]
            if not match.empty:
                secili_musteri = match.iloc[0].to_dict()
                ad_display = mask_text(secili_musteri.get('Ad_Soyad', 'Bilinmeyen Müşteri')) if maskeleme_aktif else secili_musteri.get('Ad_Soyad', 'Bilinmeyen Müşteri')
                musteri_segmenti = musteri_segment_belirle(secili_musteri.get('Sadakat_Puani'))
                st.success(f"Müşteri Bulundu: **{ad_display}** ({musteri_segmenti} Segment)")
            else:
                st.info("ℹ️ Kayıtlı müşteri bulunamadı. Genel müşteri profili ile devam ediliyor.")

        st.divider()

        harcanan_tutar = st.number_input(
            "💰 Anlık Sepet Tutarı (TL):",
            min_value=10.0,
            max_value=10000.0,
            value=350.0,
            step=10.0,
            key="input_tutar",
            on_change=reset_analysis
        )
        sepetteki_urun = st.slider(
            "📦 Sepetteki Ürün Adedi:",
            min_value=1,
            max_value=20,
            value=3,
            key="input_urun",
            on_change=reset_analysis
        )

        kategoriler = list(KATEGORI_KOLON_ONEKI.keys())
        secilen_kategori = st.selectbox(
            "🏷️ Sepetteki Ağırlıklı Kategori:",
            kategoriler,
            key="input_kat",
            on_change=reset_analysis
        )

        if st.button("⚡ AI Önerilerini ve Tahmini Hesapla", type="primary", use_container_width=True):
            st.session_state.analiz_yapildi = True
            st.session_state.last_feedback_msg = None
            st.session_state.last_feedback_type = None

    with col_right:
        st.subheader("2. Yapay Zeka & Kasa Fırsat Analizi")

        if st.session_state.analiz_yapildi:
            if model is None:
                st.error("Model yüklü olmadığı için tahmin yapılamıyor.")
            else:
                # --- A. FEATURE VEKTÖRÜNÜ HAZIRLAMA (53 ÖZELLİKLİ MODEL MİMARİSİ) ---
                # Eşleşen müşteri varsa onun GERÇEK geçmiş davranış profiliyle,
                # yoksa veri setinin genel (medyan/mod) profiliyle başlıyoruz;
                # böylece "genel müşteri profili" mesajı gerçekten doğru olur
                # (sıfırlarla değil).
                oneri_urun, oneri_fiyat = KATEGORI_URUN_ONERI.get(secilen_kategori, ("Kasa Önü Minis Ürünler", 19.90))

                input_dict = dict(secili_musteri) if secili_musteri else dict(GENEL_MUSTERI_PROFILI)
                input_dict["Sepet_Tutari_TL"] = harcanan_tutar
                input_dict["Onerilen_Kategori"] = secilen_kategori
                input_dict["Onerilen_Urun"] = oneri_urun

                df_input = pd.DataFrame([input_dict])

                expected_features = getattr(model, "feature_names_in_", None)
                if expected_features is not None:
                    df_input = df_input.reindex(columns=list(expected_features), fill_value=0)

                # --- B. AI TAHMİNİ ÇALIŞTIRMA ---
                try:
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(df_input)[0][1]
                    else:
                        pred = model.predict(df_input)[0]
                        proba = 0.85 if pred == 1 else 0.35

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
                except Exception as e:
                    st.warning(f"Model tahmini sırasında uyarı: {str(e)}.")
                    proba = 0.50

                st.divider()

                # --- C. ÖZELLİK 1: AÇIKLANABİLİR YAPAY ZEKA (XAI) ---
                st.markdown("#### 🔍 Yapay Zeka Karar Gerekçeleri")
                xai_list = generate_xai_insights(secili_musteri, harcanan_tutar, secilen_kategori, proba)

                box_style = "xai-box" if proba >= 0.5 else "xai-box-low"
                st.markdown(f'<div class="{box_style}">', unsafe_allow_html=True)
                for item in xai_list:
                    st.markdown(f"- {item}")
                st.markdown('</div>', unsafe_allow_html=True)

                st.divider()

                # --- D. AKILLI ÇAPRAZ SATIŞ & UPSELL ÖNERİ MOTORU ---
                st.markdown("#### 💡 Kasiyer İçin Anlık Aksiyon Önerileri")

                if harcanan_tutar < 500:
                    kalan = 500 - harcanan_tutar
                    st.info(f"🎁 **Kampanya Fırsatı:** Sepete **{kalan:.2f} TL** daha eklendiğinde 50 TL Kasa İndirimi kazanılıyor!")
                elif harcanan_tutar >= 500 and harcanan_tutar < 1000:
                    st.success("🎉 Müşteri 500 TL üzeri kargo/indirim limitine ulaştı! VIP hediye çeki sunulabilir.")

                st.warning(f"👉 **Önerilen Kasa Önü Ürünü:** {oneri_urun} — **Özel Fiyat:** {oneri_fiyat:.2f} TL")

                st.divider()

                # --- E. ÖZELLİK 3: CANLI GERİ BİLDİRİM DÖNGÜSÜ (FEEDBACK LOOP) ---
                st.markdown("#### 🔄 Müşteri Yanıtı Kaydı (Feedback Loop)")
                st.caption("Kasiyer teklifi sunduktan sonra müşterinin kararını kaydedin. Bu veriler MLOps & ROI analizini besler.")

                # Kalıcı görsel bildirim kutusu
                if st.session_state.last_feedback_msg:
                    if st.session_state.last_feedback_type == "success":
                        st.success(st.session_state.last_feedback_msg)
                    else:
                        st.info(st.session_state.last_feedback_msg)

                fb_col1, fb_col2 = st.columns(2)

                with fb_col1:
                    if st.button("✅ Müşteri Öneriyi Kabul Etti", use_container_width=True, type="secondary"):
                        st.session_state.total_recommendations += 1
                        st.session_state.accepted_recommendations += 1
                        st.session_state.ai_generated_revenue += oneri_fiyat
                        st.session_state.last_feedback_msg = f"🎉 **Kabul Kaydedildi:** Sepete +{oneri_fiyat:.2f} TL eklendi! Sol menüdeki ROI metrikleri güncellendi."
                        st.session_state.last_feedback_type = "success"
                        st.rerun()

                with fb_col2:
                    if st.button("❌ Müşteri Öneriyi Reddetti", use_container_width=True):
                        st.session_state.total_recommendations += 1
                        st.session_state.rejected_recommendations += 1
                        st.session_state.last_feedback_msg = "ℹ️ **Red Kaydedildi:** Yanıt model iyileştirme veri havuzuna aktarıldı."
                        st.session_state.last_feedback_type = "info"
                        st.rerun()

        else:
            st.info("👈 Önerileri ve XAI detaylarını görmek için lütfen sol taraftaki **'AI Önerilerini ve Tahmini Hesapla'** butonuna basın.")

# ------------------------------------------
# TAB 2: MÜŞTERİ & MAĞAZA ANALİTİĞİ
# ------------------------------------------
with tab_analitik:
    st.subheader("📊 Müşteri Veri Seti & Mağaza Özet Analizleri")

    st.markdown("### 💰 AI Katma Değer & İş Etkisi")

    roi1, roi2, roi3, roi4 = st.columns(4)
    roi1.metric("Toplam Sunulan Öneri", st.session_state.total_recommendations)
    roi2.metric("Kabul Edilen Öneri", st.session_state.accepted_recommendations)

    conv_rate = (st.session_state.accepted_recommendations / st.session_state.total_recommendations * 100) if st.session_state.total_recommendations > 0 else 0.0
    roi3.metric("Kasa Dönüşüm Oranı", f"%{conv_rate:.1f}")
    roi4.metric("AI Kaynaklı Ek Ciro", f"{st.session_state.ai_generated_revenue:.2f} TL")

    st.divider()

    if not df_musteriler.empty:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Toplam Kayıtlı Müşteri", len(df_musteriler))

        # Segment sütunu bu veri setinde yok; VIP tanımı Sadakat_Puani'nın
        # üst %25'i olarak (yukarıda VIP_ESIK_DEGERI ile) türetiliyor.
        if "Sadakat_Puani" in df_musteriler.columns:
            vip_count = int((df_musteriler["Sadakat_Puani"] >= VIP_ESIK_DEGERI).sum())
        else:
            vip_count = 0
        col_m2.metric("VIP Müşteri Sayısı", vip_count)

        # Eski 'gecmis_ortalama_sepet_tutari' sütununun karşılığı: Sepet_Tutari_TL
        col_m3.metric("Ort. Harcama Tutarı", f"{df_musteriler['Sepet_Tutari_TL'].mean():.1f} TL" if 'Sepet_Tutari_TL' in df_musteriler else "N/A")
        # Eski 'musteri_kidem_gun' (kayıt tarihi) bu veri setinde yok; en yakın
        # anlamlı karşılığı genel son-aktivite göstergesi olan Gecen_Gun_Sayisi.
        col_m4.metric("Ort. Son Aktivite (Gün)", f"{df_musteriler['Gecen_Gun_Sayisi'].mean():.0f} Gün" if 'Gecen_Gun_Sayisi' in df_musteriler else "N/A")

        st.divider()
        st.markdown("##### 📄 Mevcut Müşteri Veri Tablosu")

        df_display = df_musteriler.copy()
        if maskeleme_aktif and 'Ad_Soyad' in df_display.columns:
            df_display['Ad_Soyad'] = df_display['Ad_Soyad'].apply(lambda x: mask_text(str(x)))

        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.warning("Görüntülenecek müşteri verisi bulunamadı.")

# ------------------------------------------
# TAB 3: YENİ MÜŞTERİ KAYDI
# ------------------------------------------
with tab_yeni_musteri:
    st.subheader("➕ Yeni Müşteri Ekleme Formu")
    with st.form("yeni_musteri_formu", clear_on_submit=True):
        f_ad = st.text_input("Ad Soyad:")
        f_tel = st.text_input("Telefon No:")
        f_segment = st.selectbox("Segment:", ["Yeni Müşteri", "Standart Müşteri", "Sadık Müşteri", "VIP Müşteri"])

        submit_btn = st.form_submit_button("Müşteriyi Kaydet")
        if submit_btn:
            if f_ad and f_tel:
                st.success(f"✅ Müşteri `{f_ad}` başarıyla eklendi! (Simülasyon)")
            else:
                st.error("Lütfen Ad Soyad ve Telefon alanlarını doldurun.")
