(ns
  ^{:author abc}
   testdev.view.contents)


;http://getbootstrap.com/components/#navbar
;http://www.bootply.com/EY5Gi2T0eD

(defn fixed-navigation-bar []
  [:nav {:class "navbar navbar-custom navbar-inverse navbar-static-top" :id "nav"}
   [:div {:class "container"}
    [:div {:class "navbar-header"}
     [:input {:type "button" :class "navbar-toggle collapsed" :data-toggle "collapse" :data-target ".navbar-collapse" :aria-expanded "false"}
      [:span {:class "icon-bar"}]
      [:span {:class "icon-bar"}]
      [:span {:class "icon-bar"}]
      ]
     [:a {:class "navbar-brand" :href "#"} "Brand"]
     ]
    [:div {:class "collapse navbar-collapse" :id "bs-example-navbar-collapse-1"}
     [:ul {:class "nav navbar-nav"}
      [:li
       [:a {:href "#"} "Home"]]
      [:li {:role "separator" :class "divider"}]
      [:li
       [:a {:href "#section2"} "Product"]]
      [:li {:role "separator" :class "divider"}]
      [:li
       [:a {:href "#section3"} "News"]]
      [:li {:role "separator" :class "divider"}]
      [:li {:class "active"}
       [:a {:href "#section1"} "Big Brand"]]
      [:li {:role "separator" :class "divider"}]
      [:li {:role "separator" :class "divider"}]
      [:li
       [:a {:href "#section4"} "About"]]
      [:li {:role "separator" :class "divider"}]
      [:li
       [:a {:href "#section5"} "Contact"]]
      [:li {:role "separator" :class "divider"}]
      [:li {:class "dropdown"}
       [:a {:href "#section8" :class "dropdown-toggle" :data-toggle "dropdown" :role "button"
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
        ]
       ]
      ]
     ]
    ]])

