(ns
  ^{:author abc}
   testdev.contents)


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
     "."]]]

   [:div {:class "row"}
    [:div {:class "span3"}
     [:p {:class "sans-serif"}
      "We work with individuals that create solutions, celebrate entrepreneurship, stand for justice, and believe in the progress of common good."
      ]
     [:br]
     [:br]
     [:button {:type "button" :class "btn btn-success btn-lg" :href "#"} "About our company"]]

    [:div {:class "span6"}
     [:img {:src "http://whiteboard.is/wp-content/themes/wb4/images/team-home-landing-full.jpg"}]
     ]
    ]



   ]
  )

