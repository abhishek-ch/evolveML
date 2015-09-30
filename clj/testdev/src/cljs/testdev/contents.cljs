(ns
  ^{:author abc}
   testdev.contents

 )


;http://getbootstrap.com/components/#navbar
;http://www.bootply.com/EY5Gi2T0eD

(defn fixed-navigation-bar []
   [:div {:class "container"}
    [:div {:class "navbar-header"}
     [:button {:type "button" :class "navbar-toggle" :data-toggle "collapse" :data-target ".navbar-collapse"}
      [:span {:class "icon-bar"}]
      [:span {:class "icon-bar"}]
      [:span {:class "icon-bar"}]
      ]
     ]
    [:div {:class "collapse navbar-collapse" :id "bs-example-navbar-collapse-1"}
     [:ul {:class "nav navbar-nav nav-justified"}
      [:li
       [:a {:href "#"} "Home"]]
      [:li
       [:a {:href "#section2"} "Product"]]
      [:li
       [:a {:href "#section3"} "News"]]
      [:li {:class "active"}
       [:a {:href "#section1"} "Big Brand"]]
      [:li
       [:a {:href "#section4"} "About"]]
      [:li
       [:a {:href "#section5"} "Contact"]]
      [:li {:role "separator" :class "divider"}]
      [:li {:class "dropdown"}
       [:a {:href "#section8" :class "dropdown-toggle" :data-toggle "dropdown"
            :aria-haspopup "true" :aria-expanded "false"} "More"
        [:span {:class "caret"}]]
       [:ul {:class "dropdown-menu"}
        [:li
         [:a {:href "#"} "Action"]]
        [:li
         [:a {:href "#"} "Testing Action"]]
        [:li
         [:a {:href "#"} "Just let it go"]]
        [:li
         [:a {:href "#"} "Astounding"]]
        [:li
         [:a {:href "#"} "Get me some link"]]
        ]]
      ]]])



(defn pagecontents []
  [:div {:class "col-sm-10 col-sm-offset-1"}
   [:div {:class "page-header text-center"}
    [:h1 "Debdoot Agency is Known for there Medical Identity Since 30 years"]
    ]
   [:p {:class "lead text-center"}
    "We.. are Wholesellers Dealers and One Stop shop for anything with medicines."
    ]
   [:p {:class "text-center"}
    "Most of the EMA's scientific evaluation work is carried out by its scientific committees,
    which are made up of members from EEA countries, as well as representatives of patient, consumer
    and healthcare-professional organisations. These committees have various tasks related to the development,
    assessment and supervision of medicines in the EU"
    ]
   ]
 )


(defn pageWithImageOne []
   [:div {:class "col-sm-6 col-sm-offset-3 text-center"}
    [:h2 "Debdoot Agency , Distributor .. Wholeseller"]
    ])


(defn page2 []
  [:div {:class "col-sm-8 col-sm-offset-2 text-center"}
   [:h1 "We Do things Differently"]
   [:blockquote
    [:span
     "We are a creative company build to "
     ]
    [:span {:class "hover-active"}
     "purposeful ventures"
     ]
    ]
   ]
  )


;http://whiteboard.is/
;http://getbootstrap.com/2.3.2/scaffolding.html
(defn pageOwn []
  [:div {:class "container"}
  [:div {:class "row"}
   [:div {:class "twelve columns"}
    [:blockquote
     [:span
      "We are a creative company build to "]
     [:span {:class "hover-active"}
      "purposeful ventures"]
     ". We are led by our desire to produce "
     [:span {:class "hover-active"}
      " Love for life "]
     ","
     [:span {:class "hover-active"} "people "]
     ", and "
     [:span {:class "hover-active"} " possibilities "]
     ". That's why we are well known as "
     [:span {:class "hover-active"} " Debdoot Agency "]
     ]]
     ]

   [:div {:class "row"}
    [:div {:class "span3"}
     [:p {:class "sans-serif"}
      "We work with individuals that create solutions, celebrate entrepreneurship, stand for justice, and believe in the progress of common good."
      ]
     [:br]
     [:button {:type "button" :class "btn btn-success btn-lg" :href "#"} "About our company"]]
    [:br]
    [:br]

   ]]
  )


(defn explainDelivery []
 [:div {:class "col-sm-6 col-sm-offset-3 text-center"}
  [:h2 {:style {:padding "50px" :background-color "rgba(5,5,5,.8)"}}
   "Distributor -> [BadaBazar & Aamtala]"
   [:br]
   "Wholeseller -> [Badabazar aka. Shankarnarayan]"
   ]]
 )


(defn pictureBlock
 [src leadVal content]
 [:div {:class "col-sm-4 col-xs-6"}
 [:div {:class "panel panel-default"}
  [:div
   [:img {:src src :class "img-responsive"}]]
  [:div {:class "panel-body"}
   [:p {:class "lead"} leadVal]
   [:p content]
   ]
  ]]
 )

(def contactsMap
 [{:src "/img/1.jpg" :lead "Anup Saha" :lead_content "Owner/Founder"}
  {:src "/img/2.jpg" :lead "Radha Saha" :lead_content "Co-Founder & Wife"}
  {:src "/img/2.jpg" :lead "Thinking..." :lead_content "Advisory Board"}
  ]
 )


(defn delegatePictureBlock []
 (map #(pictureBlock (:src %1)(:lead %1)(:lead_content %1)) contactsMap)
 )


(defn get-places []
 (let [map-canvas (.getElementById js/document "map-canvas")
       map-options (clj->js {"center" (google.maps.LatLng. 22.381742, 88.270003)
                             "zoom" 8})]
  (js/google.maps.Map. map-canvas map-options))
 )


(defn formDiv []

[:div {:style {:padding "50px"}}
 [:div {:class "col-sm-8"}
  [:div {:class "row form-group"}
   [:div {:class "col-xs-3"}
    [:input {:type "text" :class "form-control" :id "firstName" :name "firstName" :placeholder "First Name" :required ""}]]
   [:div {:class "col-xs-3"}
    [:input {:type "text" :class "form-control" :id "middleName" :name "middleName" :placeholder "Middle Name" :required ""}]]
   [:div {:class "col-xs-4"}
    [:input {:type "text" :class "form-control" :id "lastName" :name "lastName" :placeholder "Last Name" :required ""}]]
   ]

  [:div {:class "row form-group"}
   [:div {:class "col-xs-5"}
    [:input {:type "email" :class "form-control" :name "email" :placeholder "Email" :required ""}]]
   [:div {:class "col-xs-5"}
    [:input {:type "phone" :class "form-control" :name "phone" :placeholder "Phone" :required ""}]]
   ]

  [:div {:class "row form-group"}
   [:div {:class "col-xs-10"}
    [:input {:type "homepage" :class "form-control" :placeholder "Website URL" :required ""}]]
   ]


  [:div {:class "row form-group"}
   [:div {:class "col-xs-10"}
    [:button {:class "btn btn-default pull-right"} "Contact Us"]]]
  ]

 [:div {:class "col-sm-3 pull-right"}
  [:address
   [:strong "Debdoot Agency"] [:br]
   " 44 Ezra Street, Gandhi Building"[:br]
   "3rd Floor Room No - 14 & 15 Bada Bazar"[:br]
   "Kolkata - 700001"[:br]
   "West Bengal, India"[:br]
   "P(h) - (+91)8017 26 7718"
   ]
  [:address
   [:strong "Email Us "]
   [:a {:href "mailto:#"} "debdoot.agency.amtala@gmail.com"]
   ]]]

  )

(defn direction []
 ;AIzaSyBVe4qpNwgKVUU_g62OSqpXi6H6FDp5UyU
 ;22.381742, 88.270003
[:div {:style {:padding "50px"}}
 [:h1 {:class "text-center"} "Our Location"]
 [:div {:id "map-canvas"}]
 [:hr]
  (formDiv)
])


(defn beforefooter []
 [:div {:class "col-sm-8 col-sm-offset-2 text-center"}
  [:h2 "Debdoot Agency Group"]
  [:hr]
  [:h4 "Distributor & Wholeseller"]
  [:p
   "Connect or Contact us for knowing about our clients and Nature of Business"
   ]
  ])


(defn footer []
 [:div {:class "container"}
  [:p {:class "text-muted"} "Developer by ABC aka Abhishek Choudhary "[:a {:href "https://abhishek-choudhary.blogspot.in"} "[Blog] "
                                                [:a {:href "https://github.com/abhishek-ch"} " [github]"]] ]
  ]
 )

(defn pull-up []
 [:li
  [:a {:href "#" :title "Scroll to top"}
   [:i {:class "glyphicon glyphicon-chevron-up"}]
   ]
  ]
 )





