(ns
  ^{:author abc}
  myfirstclj.views.recontents
  (:use [reagent.core])
  (:require [myfirstclj.views.layout :as layout]
            [reagent.core :as reagent]
            [reagent-modals.modals :as reagent-modals]))


(defn fixed-navigation-bar []
  [:div {:class "navbar navbar-custom navbar-inverse navbar-static-top" :id "nav"}
   [:div {:class "container"}
    [:div {:class "navbar-header"}
     [:input {:type "button" :class "navbar-toggle" :data-toggle "collapse" :data-target ".navbar-collapse"}
      [:span {:class "icon-bar"}]
      [:span {:class "icon-bar"}]
      [:span {:class "icon-bar"}]
      ]
     ]
    [:div {:class "collapse navbar-collapse"}
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
      ]
     ]

    ]
   ]
  )


(defn not-found []
  [:div
   [:h1 {:class "info-worning"} "Page Not Found"]
   [:p "There's no requested page. "]
   (link-to {:class "btn btn-primary"} "/" "Take me to Home")])


(defn index []
  (layout/application "Reagent - Search..."
    (fixed-navigation-bar)
    [:div {:class "clear"}]
    ))