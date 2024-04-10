#define glx__sim
#include "DrcEvent.h"
#include "glxtools.C"

struct EventStruct{
    int EventID;
    int PDG;
    int NHits;
    int BarID;
    Float_t invMass;
    Float_t P;
    Float_t Theta;
    Float_t Phi;
    Float_t X;
    Float_t Y;
    Float_t Z;
    vector <int> pmtID;
    vector <int> pixelID;
    vector <int> channel;
    vector <float> pos_x;
    vector <float> pos_y;
    vector <float> pos_z;
    vector <float> leadTime;
} ;

void WriteStructJson(EventStruct evtstruct, std::ofstream& out)
{
    out << "{" <<
    Form("\"EventID\" : %d, ", evtstruct.EventID) <<
    Form("\"PDG\" : %d, ", evtstruct.PDG) <<
    Form("\"NHits\" : %d, ", evtstruct.NHits) <<
    Form("\"BarID\" : %d, ", evtstruct.BarID) <<
    Form("\"invMass\" : %f, ", evtstruct.invMass) <<
    Form("\"P\" : %f, ", evtstruct.P) <<
    Form("\"Theta\" : %f, ", evtstruct.Theta) <<
    Form("\"Phi\" : %f, ", evtstruct.Phi) <<
    Form("\"X\" : %f, ", evtstruct.X) <<
    Form("\"Y\" : %f, ", evtstruct.Y) <<
    Form("\"Z\" : %f, ", evtstruct.Z) <<
    Form("\"pmtID\" : [");
    for (int i = 0; i < evtstruct.NHits; i++)
    {
        out << evtstruct.pmtID[i];
        if (i < evtstruct.NHits - 1)
        {
            out << ", ";
        }
    }
    out << "], " <<
    Form("\"pixelID\" : [");
    for (int i = 0; i < evtstruct.NHits; i++)
    {
        out << evtstruct.pixelID[i];
        if (i < evtstruct.NHits - 1)
        {
            out << ", ";
        }
    }   
    out << "], " <<
    Form("\"channel\" : [");
    for (int i = 0; i < evtstruct.NHits; i++)
    {
        out << evtstruct.channel[i];
        if (i < evtstruct.NHits - 1)
        {
            out << ", ";
        }
    }
    out << "], " <<
    Form("\"leadTime\" : [");
    for (int i = 0; i < evtstruct.NHits; i++)
    {
        out << evtstruct.leadTime[i];
        if (i < evtstruct.NHits - 1)
        {
            out << ", ";
        }
    }
    out << "]}" << endl;
}

void MakeDictionaries(TString inFileName, TString outFileName = "Phi.json")
{
    if(!glx_initc(inFileName,1,"data/drawHP")) return;
    const int nEvents = glx_ch->GetEntries();
    cout << "Total number of Entries found : " << nEvents << endl;
    int pion_counts = 0;
    int kaon_counts = 0;
    ofstream outFile;
    outFile.open(outFileName.Data());
    //outFile << "{";
    
    for (int ev = 0; ev < nEvents; ev++)
    {
        glx_ch->GetEntry(ev);
        const int ps = glx_events->GetEntriesFast();
        if (ps > 2)
        {
           //cout << "Skipping because p " << ps << endl;
           continue; // Why is this ??
        }
        for (int p = 0; p < ps; p++)
        {
            //cout << "EV: " << ev << ": " << p << endl;
            //if (p != 0) continue;
            
            glx_nextEventc(ev,p,10);

            // if(glx_event->GetParent()>0) continue;
            // //  cout << "mass: " << glx_event->GetInvMass() << " PDG" << glx_event->GetPdg() << endl;
            // // For Rho -> pi+pi- 776 +- 147 MeV
            // if (abs(glx_event->GetPdg()) == 211)
            // {
            //     if ((glx_event->GetInvMass() < 0.776 - 0.147) || (glx_event->GetInvMass() > 0.776 + 0.147))
            //     {   //cout << "Skipping PDG " << evtStruct.PDG << " with Mass " << evtStruct.invMass << endl;
            //         continue;
            //     }
            // }
            // // For phi -> K+K- 1019 +- 10 MeV
            // else if (abs(glx_event->GetPdg()) == 321)
            // {
            //     if ((glx_event->GetInvMass() < 1.0) || (glx_event->GetInvMass() > 1.1))
            //     {   //cout << "Skipping PDG " << evtStruct.PDG << " with Mass " << evtStruct.invMass << endl;
            //         continue;
            //     }
            // }
            
            // if ((abs(glx_event->GetPdg()) != 321) || (abs(glx_event->GetPdg()) != 211)){
            //      cout << "Skipping: " <<  glx_event->GetPdg() << endl;
            //      //cout << glx_event->GetPdg() << endl;
            //      continue;}

            EventStruct evtStruct;
            evtStruct.EventID = ev;
            evtStruct.PDG = glx_event->GetPdg();
            evtStruct.NHits = glx_event->GetHitSize();
            evtStruct.BarID = glx_event->GetId();
            evtStruct.P = glx_event->GetMomentum().Mag();
            evtStruct.Theta = glx_event->GetMomentum().Theta()*180/TMath::Pi();
            evtStruct.Phi = glx_event->GetMomentum().Phi()*180/TMath::Pi();
            TVector3 hpos = glx_event->GetPosition();
            evtStruct.X = hpos.X();
            evtStruct.Y = hpos.Y();
            evtStruct.Z = hpos.Z();
            evtStruct.invMass = glx_event->GetInvMass();
            
            for (int h = 0; h < evtStruct.NHits; h++)
            {
                DrcHit hit = glx_event->GetHit(h);
                evtStruct.pmtID.push_back(hit.GetPmtId());
                evtStruct.pixelID.push_back(hit.GetPixelId());
                evtStruct.channel.push_back(hit.GetChannel());
                evtStruct.pos_x.push_back(hit.GetPosition().X());
                evtStruct.pos_y.push_back(hit.GetPosition().Y());
                evtStruct.pos_z.push_back(hit.GetPosition().Z());
                evtStruct.leadTime.push_back(hit.GetLeadTime());
            }

            WriteStructJson(evtStruct, outFile);
        }
    }
    outFile.close();
}
